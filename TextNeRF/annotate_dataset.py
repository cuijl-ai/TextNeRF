import torch
import numpy as np
import os
import json
from tqdm import tqdm
from PIL import Image
# data
from datasets.textnerf import TextNeRFSynthesisDataset

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import XYZEncoder, ColorRenderer_G, TextNGP_G

from misc.utils import load_ckpt
from misc.camera import TextNeRFEngine
from misc.text_pose import visualize_text_pose, visualize_text_anno

import warnings;
warnings.filterwarnings("ignore")
np.random.seed(123)


def prepare_scene(scale, image_root_dir, downsample, device):
    dataset = TextNeRFSynthesisDataset(root_dir=image_root_dir, downsample=downsample)
    xyz_encoder = XYZEncoder(scale=scale)
    color_renderer = ColorRenderer_G()
    model = TextNGP_G(scale=scale, num_images=dataset.training_embed_size, xyz_encoder=xyz_encoder, color_renderer=color_renderer)
    G = model.grid_size
    model.register_buffer('density_grid', torch.zeros(model.cascades, G ** 3))
    model.register_buffer('grid_coords', create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))
    return dataset, model.to(device)
    

def vis_annotation(K, label_file, root_dir, save_dir, box_only=True):
    os.makedirs(save_dir, exist_ok=True)
    with open(label_file, encoding="utf-8") as f:
        for line in tqdm(f.readlines(), desc="Visualizing annos"):
            line = line.strip()
            if line:
                basename, annos = line.split('\t')
                src_image_path = os.path.join(root_dir, "images", basename)
                pil_image = Image.open(src_image_path)
                annos = json.loads(annos)
                dst_image_path = os.path.join(save_dir, os.path.splitext(basename)[0]+".jpg")
                try:
                    if box_only:
                        vis_pil_image = visualize_text_anno(annos, pil_image)
                    else:
                        vis_pil_image = visualize_text_pose(annos, K, pil_image, pose_only=False, scale=1.0)
                    vis_pil_image.save(dst_image_path)
                except:
                    print(f"Visualizing {src_image_path} error!!!")
         
         
def process_one_scene(root_dir, ckpt_path, hand_anno_path, output_dir, all_txt_on_same_plane,
                      scale=4, downsample=1.0, device='cuda'):
    dataset, ngp = prepare_scene(scale, root_dir, downsample, device)
    load_ckpt(ngp, ckpt_path)
    text_nerf = TextNeRFEngine(ngp, dataset, all_txt_on_same_plane=all_txt_on_same_plane)
    
    os.makedirs(output_dir, exist_ok=True)
    annotation_save_path = os.path.join(output_dir, 'annotations.txt')
    text_nerf.annotate_dataset(hand_anno_path, annotation_save_path)
    
    vis_result_dir = os.path.join(output_dir, "vis_annos")
    os.makedirs(vis_result_dir, exist_ok=True)
    vis_annotation(text_nerf.K, annotation_save_path, root_dir, vis_result_dir)
    
    meta_data_path = os.path.join(output_dir, "meta_data.json")
    meta_data = dict(camera_intrinsic=text_nerf.K.tolist(), 
                     image_wh=text_nerf.img_wh)
    for image_path, pose in zip(text_nerf.dataset.img_paths, text_nerf.dataset.poses):
        meta_data[os.path.basename(image_path)] = {"pose": pose.tolist()}
    with open(meta_data_path, "w", encoding='utf-8') as f:
        json.dump(meta_data, f, ensure_ascii=False)
    

if __name__ == '__main__':
    name = ...  # the name of the trained scene
    root_dir = ...  # dataset of scene image root
    ckpt_path = ...  # ckpt for trained nerf model
    hand_anno_path = ...  # pre-annotated labels
    output_dir = ...  # output dir
    process_one_scene(root_dir, ckpt_path, hand_anno_path, output_dir, all_txt_on_same_plane=False)
