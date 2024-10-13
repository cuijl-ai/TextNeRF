import torch
from torch import nn
from einops import rearrange
from .ssim import SSIM
import vren


class DistortionLoss(torch.autograd.Function):
    """
    Distortion loss proposed in Mip-NeRF 360 (https://arxiv.org/pdf/2111.12077.pdf)
    Implementation is based on DVGO-v2 (https://arxiv.org/pdf/2206.05085.pdf)

    Inputs:
        ws: (N) sample point weights
        deltas: (N) considered as intervals
        ts: (N) considered as midpoints
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]

    Outputs:
        loss: (N_rays)
    """
    @staticmethod
    def forward(ctx, ws, deltas, ts, rays_a):
        loss, ws_inclusive_scan, wts_inclusive_scan = \
            vren.distortion_loss_fw(ws, deltas, ts, rays_a)
        ctx.save_for_backward(ws_inclusive_scan, wts_inclusive_scan,
                              ws, deltas, ts, rays_a)
        return loss

    @staticmethod
    def backward(ctx, dL_dloss):
        (ws_inclusive_scan, wts_inclusive_scan,
        ws, deltas, ts, rays_a) = ctx.saved_tensors
        dL_dws = vren.distortion_loss_bw(dL_dloss, ws_inclusive_scan,
                                         wts_inclusive_scan,
                                         ws, deltas, ts, rays_a)
        return dL_dws, None, None, None


class S3IM(torch.nn.Module):
    r"""Implements Stochastic Structural SIMilarity(S3IM) algorithm.
    It is proposed in the ICCV2023 paper
    `S3IM: Stochastic Structural SIMilarity and Its Unreasonable Effectiveness for Neural Fields`.

    Arguments:
        kernel_size (int): kernel size in ssim's convolution(default: 4)
        stride (int): stride in ssim's convolution(default: 4)
        repeat_time (int): repeat time in re-shuffle virtual patch(default: 10)
        patch_height (height): height of virtual patch(default: 64)
        patch_width (height): width of virtual patch(default: 64)
    """
    def __init__(self, kernel_size=4, stride=4, repeat_time=10, patch_height=64, patch_width=64):
        super(S3IM, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        # self.repeat_time = repeat_time
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.ssim_loss = SSIM(window_size=self.kernel_size, stride=self.stride)

    def forward(self, src_vec, tar_vec):
        loss = 0.0
        index_list = []
        for i in range(1, 100):
            if src_vec.shape[0] * i % (self.patch_height * self.patch_width) == 0:
                auto_repeat_time = i
                break
        for i in range(auto_repeat_time):
            if i == 0:
                tmp_index = torch.arange(len(tar_vec))
                index_list.append(tmp_index)
            else:
                ran_idx = torch.randperm(len(tar_vec))
                index_list.append(ran_idx)
        res_index = torch.cat(index_list)
        tar_all = tar_vec[res_index]
        src_all = src_vec[res_index]
        tar_patch = tar_all.permute(1, 0).reshape(1, 3, self.patch_height, -1)
        src_patch = src_all.permute(1, 0).reshape(1, 3, self.patch_height, -1)
        loss = (1 - self.ssim_loss(src_patch, tar_patch))
        return loss


class AppearanceStageLoss(nn.Module):
    def __init__(self, lambda_appearance=1e-3, text_area_enhance_ratio=1.2,
                 appearance_l2_regular=1e-5, mode_seeking_weight=1e-6, color_mode="RGB"):
        super().__init__()
        self.lambda_appearance = lambda_appearance
        self.text_area_enhance_ratio = text_area_enhance_ratio
        self.appearance_l2_regular = appearance_l2_regular
        self.mode_seeking_weight = mode_seeking_weight
        self.color_mode = color_mode

    def forward(self, results, target):
        d = {}
        if self.color_mode == "RGB":
            d['color'] = (results['color'] - target['color']) ** 2
        elif self.color_mode == "LAB":
            d['color'] = (results['color'] - target['color']) ** 2
            d['color'][..., 0] = d['color'][..., 0] * ((100 / 255) ** 2)

        d['a_regular'] = self.appearance_l2_regular * torch.pow(results['appearance_embedding'], 2)
        d['appearance'] = self.lambda_appearance * \
                          torch.abs(results['appearance_embedding'] - target['appearance_embedding'].detach())
        if target.get("text_sem_map", None) is not None:
            cof = torch.where(target["text_sem_map"] > 0,  self.text_area_enhance_ratio, 1 / self.text_area_enhance_ratio)
            cof = rearrange(cof, 'h w -> (h w) 1')
            d['color'] = d['color'] * cof

        return d


class GeometryStageLoss(nn.Module):
    def __init__(self, lambda_opacity=1e-3, lambda_distortion=1e-3, lambda_s3im=1.0, color_mode="RGB"):
        super().__init__()

        self.lambda_opacity = lambda_opacity
        self.lambda_distortion = lambda_distortion
        self.lambda_s3im = lambda_s3im
        if self.lambda_s3im > 0:
            self.s3im = S3IM()
        self.color_mode = color_mode

    def forward(self, results, target, **kwargs):
        d = {}
        if self.color_mode == "RGB":
            d['color'] = (results['color'] - target['color']) ** 2
        elif self.color_mode == "LAB":
            d['color'] = (results['color'] - target['color']) ** 2
            d['color'][..., 0] = d['color'][..., 0] * ((100 / 255) ** 2)

        if self.lambda_s3im > 0:
            d['s3im'] = self.lambda_s3im * self.s3im(results['color'], target['color'])

        o = results['opacity']+1e-10
        # encourage opacity to be either 0 or 1 to avoid floater
        d['opacity'] = self.lambda_opacity*(-o*torch.log(o))

        if self.lambda_distortion > 0:
            d['distortion'] = self.lambda_distortion * \
                DistortionLoss.apply(results['ws'], results['deltas'],
                                     results['ts'], results['rays_a'])

        return d
