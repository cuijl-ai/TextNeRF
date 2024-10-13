import argparse
import yaml
import os


def dump_default_yaml(root_dir, config_path, label_txt=None, downsample=1.0, exp_name="exp",
                      ckpt_path_geometry=None, num_epochs_geometry=10, ckpt_path_appearance=None):
    config = dict(
        # dataset parameters
        root_dir=root_dir, # root directory of dataset
        label_txt=label_txt if label_txt is not None else os.path.join(root_dir, "labels", "Label.txt"),
        appearance_dir=os.path.join(root_dir, "appearance"),
        text_sem_dir=os.path.join(root_dir, "labels"),
        downsample=downsample, # downsample factor (<=1.0) for the images
        geometry_sample_ratio=0.5,
        color_mode="RGB",

        # model parameters
        scale=4, # scene scale (whole scene must lie in [-scale, scale]^3)
        appearance_embed_dim=32, # appearance control
        
        # loss parameters
        ## geometry
        opacity_loss_w=1e-3,
        distortion_loss_w=1e-3, # weight of distortion loss; 0 to disable (default), to enable, a good value is 1e-3 for real scene and 1e-2 for synthetic scene
        s3im_loss_w=1.0,

        ## appearance
        appearance_loss_w=1e-3,
        text_area_enhance_ratio=1.2,
        appearance_l2_regular=1e-5,
        appearance_mode_seeking_w=1e-6,
        appearance_wh=[224, 224],

    # training options
        ## geometry stsage
        geometry_train_batch_size=8192, # number of rays in a batch
        geometry_ray_sampling_strategy='all_images',    # all_images: uniformly from all pixels of ALL images (normally for geometry stage);
                                               # same_image: uniformly from all pixels of a SAME image (normally for appearance stage).
        num_epochs_geometry=num_epochs_geometry, # number of training epochs
        lr_geometry=1e-2,  # learning rate for geometry stage
        check_val_every_n_epoch_geometry=5,
        check_val_every_n_epoch_appearance=2,

        ## appearance stage
        train_appearance=False,
        appearance_chunk=4096,
        num_epochs_appearance=10,
        lr_appearance=1e-3, # learning rate for appearance stage
        local_window_size=4,

        # experimental training options
        optimize_ext=False, # whether to optimize extrinsics
        random_bg=False, # whether to train with random bg color (real scene only) to avoid objects with black color to be predicted as transparent
    
        # validation options
        eval_lpips=False, # evaluate lpips metric (consumes more VRAM)
        no_save_test_geometry=False,  # whether to save test image and video
        no_save_test_appearance=False,
        
        # misc
        num_gpus=1,  # number of gpus
        num_sanity_val_steps=5,
        exp_name=exp_name, # experiment name
        train_results_dir="train_results",  # output results of training
        val_results_dir="val_results",
        synthesis_results_dir="synthesis_results",
        ckpt_path_geometry=ckpt_path_geometry, # pretrained checkpoint to load (including optimizers, etc)
        ckpt_path_appearance=ckpt_path_appearance,
        weight_path_geometry=None, # pretrained checkpoint to load (excluding optimizers, etc)
        weight_path_appearance=None,
    )
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        f.write(yaml.dump(config))
    print(f"Save default config YAML file to {config_path}.")


def load_config(yaml_path):
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        hparams = argparse.Namespace(**config)
    return hparams
    
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c",type=str, required=True)
    parser.add_argument("--device", "-d", type=str, default="cuda:0")
    return parser.parse_args()


def get_hparams():
    args = get_args()
    namespace = load_config(args.config_path)
    namespace.device = args.device
    return namespace
