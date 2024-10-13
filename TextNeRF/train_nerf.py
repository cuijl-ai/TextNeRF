import torch
from torch import nn
from misc.hparams import get_hparams
import os
from PIL import Image
import numpy as np
from einops import rearrange

# data
from torch.utils.data import DataLoader
from datasets.textnerf import TextNeRFGeometryDataset
from datasets.utils.ray_utils import axisangle_to_R, get_rays

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import XYZEncoder, ColorRenderer_G, TextNGP_G
from models.rendering import render, MAX_SAMPLES

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from misc.losses import GeometryStageLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

from misc.utils import slim_ckpt, load_ckpt
from misc.depth_utils import depth2img

import warnings;
warnings.filterwarnings("ignore")


class TextNeRFSystem_G(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.warmup_steps = 256
        self.update_interval = 16

        self.setup_dataset()

        self.loss = GeometryStageLoss(lambda_opacity=self.hparams.opacity_loss_w,
                                      lambda_distortion=self.hparams.distortion_loss_w,
                                      lambda_s3im=self.hparams.s3im_loss_w,
                                      color_mode=self.hparams.color_mode)

        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

        xyz_encoder = XYZEncoder(scale=self.hparams.scale)
        color_renderer = ColorRenderer_G()
        self.model = TextNGP_G(scale=self.hparams.scale, num_images=len(self.train_dataset.poses), xyz_encoder=xyz_encoder, color_renderer=color_renderer)
        G = self.model.grid_size
        self.model.register_buffer('density_grid',
                                   torch.zeros(self.model.cascades, G ** 3))
        self.model.register_buffer('grid_coords',
                                   create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))

    def forward(self, batch, split):
        if split == 'train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
            appearance_embedding = self.model.get_appearance_embedding(img_idxs=batch['img_idxs'])
        else:
            poses = batch['pose']
            directions = self.directions
            appearance_embedding = self.model.get_appearance_embedding()

        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['img_idxs']]

        rays_o, rays_d = get_rays(directions, poses)

        kwargs = {'test_time': split != 'train',
                  'random_bg': self.hparams.random_bg,
                  'appearance_embedding': appearance_embedding}
        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1 / 256

        return render(self.model, rays_o, rays_d, **kwargs)

    def setup_dataset(self):
        self.train_dataset = TextNeRFGeometryDataset(root_dir=self.hparams.root_dir,
                                                     split='train',
                                                     downsample=self.hparams.downsample,
                                                     color_mode=self.hparams.color_mode,
                                                     sample_ratio=self.hparams.geometry_sample_ratio,
                                                     batch_size=self.hparams.geometry_train_batch_size,
                                                     ray_sampling_strategy=self.hparams.geometry_ray_sampling_strategy)
        self.val_dataset = TextNeRFGeometryDataset(root_dir=self.hparams.root_dir,
                                                   split='val',
                                                   downsample=self.hparams.downsample,
                                                   color_mode=self.hparams.color_mode,
                                                   sample_ratio=self.hparams.geometry_sample_ratio)

        # define additional parameters
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

    def configure_optimizers(self):
        if self.hparams.optimize_ext:
            N = len(self.train_dataset.poses)
            self.register_parameter('dR',
                                    nn.Parameter(torch.zeros(N, 3, device=self.device)))
            self.register_parameter('dT',
                                    nn.Parameter(torch.zeros(N, 3, device=self.device)))

        load_ckpt(self.model, self.hparams.weight_path_geometry)

        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT']: net_params += [p]

        opts = []
        self.net_opt = FusedAdam(net_params, self.hparams.lr_geometry, eps=1e-15)
        opts += [self.net_opt]
        if self.hparams.optimize_ext:
            opts += [FusedAdam([self.dR, self.dT], 1e-6)]  # learning rate is hard-coded
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs_geometry,
                                    self.hparams.lr_geometry / 30)

        return opts, [net_sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)

    def on_train_start(self):
        self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
                                        self.poses,
                                        self.train_dataset.img_wh)

    def training_step(self, batch, batch_nb, *args):
        if self.global_step % self.update_interval == 0:
            self.model.update_density_grid(0.005 * MAX_SAMPLES / 3 ** 0.5,
                                           warmup=self.global_step < self.warmup_steps,
                                           erode=True)

        results = self(batch, split='train')
        loss_d = self.loss(results, batch)
        loss = sum(lo.mean() for lo in loss_d.values())

        with torch.no_grad():
            self.train_psnr(results['color'], batch['color'])
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        # ray marching samples per ray (occupied space on the ray)
        self.log('train/rm_s', results['rm_samples'] / len(batch['color']))
        # volume rendering samples per ray (stops marching when transmittance drops below 1e-4)
        self.log('train/vr_s', results['vr_samples'] / len(batch['color']))
        self.log('train/psnr', self.train_psnr, True)

        return loss

    def on_validation_start(self):
        torch.cuda.empty_cache()
        if not self.hparams.no_save_test_geometry:
            self.val_dir = f'{self.hparams.val_results_dir}/{self.hparams.exp_name}/geometry'
            os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        color_gt = batch['color']
        results = self(batch, split='test')

        logs = {}
        # compute each metric per image
        self.val_psnr(results['color'], color_gt)
        logs['psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()

        w, h = self.train_dataset.img_wh
        color_pred = rearrange(results['color'], '(h w) c -> 1 c h w', h=h)
        color_gt = rearrange(color_gt, '(h w) c -> 1 c h w', h=h)
        self.val_ssim(color_pred, color_gt)
        logs['ssim'] = self.val_ssim.compute()
        self.val_ssim.reset()
        if self.hparams.eval_lpips:
            self.val_lpips(torch.clip(color_pred * 2 - 1, -1, 1),
                           torch.clip(color_gt * 2 - 1, -1, 1))
            logs['lpips'] = self.val_lpips.compute()
            self.val_lpips.reset()

        if not self.hparams.no_save_test_geometry:  # save test image to disk
            idx = batch['img_idxs']
            color_pred = rearrange(results['color'].cpu().numpy(), '(h w) c -> h w c', h=h)
            if self.hparams.color_mode == "LAB":
                color_pred[:, :, 0] = color_pred[:, :, 0] * 100.0  # L通道反向归一化
                color_pred[:, :, 1:] = color_pred[:, :, 1:] * [255, 255] - [128, 128]  # a和b通道反向归一化
                color_pred = Image.fromarray(np.uint8(color_pred), mode='LAB').convert('RGB')
            elif self.hparams.color_mode == "RGB":
                color_pred = Image.fromarray((color_pred * 255).astype(np.uint8), mode='RGB')

            depth = Image.fromarray(depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h)), mode='RGB')

            # save results
            color_pred.save(os.path.join(self.val_dir, f'{idx:03d}.png'))
            depth.save(os.path.join(self.val_dir, f'{idx:03d}_d.png'))

        return logs

    def validation_epoch_end(self, outputs):
        psnrs = torch.stack([x['psnr'] for x in outputs])
        mean_psnr = all_gather_ddp_if_available(psnrs).mean()
        self.log('test/psnr', mean_psnr, True)

        ssims = torch.stack([x['ssim'] for x in outputs])
        mean_ssim = all_gather_ddp_if_available(ssims).mean()
        self.log('test/ssim', mean_ssim)

        if self.hparams.eval_lpips:
            lpipss = torch.stack([x['lpips'] for x in outputs])
            mean_lpips = all_gather_ddp_if_available(lpipss).mean()
            self.log('test/lpips_vgg', mean_lpips)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    hparams = get_hparams()
    # scene geometry training
    system_geometry = TextNeRFSystem_G(hparams)

    ckpt_cb_geometry = ModelCheckpoint(dirpath=f'{hparams.train_results_dir}/ckpts/{hparams.exp_name}',
                                       filename='{epoch:d}_geometry',
                                       save_weights_only=True,
                                       every_n_epochs=hparams.num_epochs_geometry,
                                       save_on_train_epoch_end=True,
                                       save_top_k=-1)
    callbacks = [ckpt_cb_geometry, TQDMProgressBar(refresh_rate=10)]
    
    logger = TensorBoardLogger(save_dir=f"{hparams.train_results_dir}/logs/geometry",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs_geometry,
                      check_val_every_n_epoch=hparams.check_val_every_n_epoch_geometry,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=DDPPlugin(find_unused_parameters=False)
                               if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=hparams.num_sanity_val_steps,
                      precision=16)
    trainer.fit(system_geometry, ckpt_path=hparams.ckpt_path_geometry)
