import os
import json
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from einops import rearrange
from tqdm import tqdm
import imageio
import cv2
import random
from PIL import Image
from skimage import measure
from collections import defaultdict, Counter
import sys
sys.path.append("..")
from datasets.utils.ray_utils import get_rays, get_ray_directions
from models.rendering import render
from datasets.utils.color_utils import read_sem_map, read_image
from .depth_utils import depth2img, save_pfm, read_pfm
from .text_pose import R_to_quaternion, get_text_pose
from .depth_utils import get_corresponding_depth


SUFFIX = {
    "img": {
        "image": ".png",
        "edited_image": ".png",
        "depth_pfm": ".pfm",
        "depth_vis": ".png",
        "sem_map": ".png",
        "text_pose_vis":  ".png",
        "custom_text_areas": ".png", 
    },
    "label": {
        "confidence": ".pkl",
        "mltview_importance": ".pkl",
        "text_labels": ".json",
        "edited_text_labels": ".json",
        "text_pose_vis":  ".png",
        "custom_text_areas": ".json",
    }
}
_EPS = np.finfo(float).eps * 4.0


def generate_bright_color(num_colors_to_generate):
    diff = lambda c1, c2: abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])
    generated_colors = []
    for _ in range(num_colors_to_generate):
        while True:
            r = random.randint(100, 255)  # 确保红色分量较大
            g = random.randint(100, 255)  # 确保绿色分量较大
            b = random.randint(100, 255)  # 确保蓝色分量较大
            if not any([diff((r, g, b), color) < 100 for color in generated_colors]):
                generated_colors.append([r, g, b])
                break
    return generated_colors


def array2rgb_img(img_array, color_mode="LAB"):
    if color_mode == "LAB":
        img_array[:, :, 0] = img_array[:, :, 0] * 100.0  # L通道反向归一化
        img_array[:, :, 1:] = img_array[:, :, 1:] * [255, 255] - [128, 128]  # a和b通道反向归一化
        rgb_img = Image.fromarray(np.uint8(img_array), mode='LAB').convert('RGB')
    elif color_mode == "RGB":
        rgb_img = Image.fromarray((img_array * 255).astype(np.uint8), mode='RGB')
    return rgb_img


def vis_custom_areas(image, custom_text_areas):
    im_array = np.asarray(image)
    # draw area box
    polygon_color = (0, 255, 0)
    polygon_points = [np.array(area["points"]).reshape((-1, 1, 2)).astype(np.int32) for area in custom_text_areas]
    cv2.polylines(im_array, polygon_points, isClosed=True, color=polygon_color, thickness=2)
    return Image.fromarray(im_array)
    

def _dump_img(photo, save_dirs, bin_colormap=None):
    SUFFIX = {
        "image": ".png",
        "edited_image": ".png",
        "depth_pfm": ".pfm",
        "depth_vis": ".png",
        "sem_map": ".png",
        "text_pose_vis":  ".png",
        "custom_text_areas": ".png", 
        "confidence": ".pkl",
        "mltview_importance": ".pkl",
    }
    for k, save_dir in save_dirs.items():
        if k not in photo:
            continue
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(photo.get("image_name", f'{photo["pose_id"]:03d}')))[0] + SUFFIX[k]
        save_path = os.path.join(save_dir, basename)
        if k in ("image", "edited_image"):
            photo[k].save(save_path)
        elif k == "depth_pfm":
            save_pfm(save_path, photo["depth"])
        elif k == "depth_vis":
            imageio.imsave(save_path, depth2img(photo["depth"]))
        elif k == "sem_map":
            sem_map = Image.fromarray(photo[k].astype(np.uint8), 'P')
            if bin_colormap is not None:
                sem_map.putpalette(bin_colormap)
            sem_map.save(save_path)
        elif k == "text_pose_vis":
            text_pose_vis = Image.fromarray(photo[k].astype(np.uint8), mode="RGB")
            text_pose_vis.save(save_path)
        elif k == "custom_text_areas":
            basename = os.path.splitext(os.path.basename(photo.get("image_name", f'{photo["pose_id"]:03d}')))[0]
            vis_save_path = os.path.join(save_dir, "vis", basename+SUFFIX[k])
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            custom_area_vis = vis_custom_areas(photo["image"], photo[k])
            custom_area_vis.save(vis_save_path)



class Camera(object):
    def __init__(self, K, img_wh, color_mode="RGB", save_dir="outputs", vis_depth=False):
        self.K = K
        self.img_wh = (w, h) = img_wh
        self.color_mode = color_mode
        self.directions = get_ray_directions(h, w, self.K)  # (wh, 3)
        self.save_dir = save_dir
        self.vis_depth = vis_depth
        assert torch.cuda.is_available()
        self.save_dirs = {
            "image": os.path.join(save_dir, "image"),
            "depth_pfm": os.path.join(save_dir, "depth_pfm")
        }
        if self.vis_depth:
            self.save_dirs["depth_vis"] = os.path.join(save_dir, "depth_vis")

    def save_rendering(self, photos):
        with ThreadPoolExecutor() as executor:
            task_pool = set()
            for photo in photos:
                photo["image"] = array2rgb_img(photo["image"], color_mode=self.color_mode)
                if hasattr(self, "bin_colormap"):
                    task_pool.add(executor.submit(_dump_img, photo, self.save_dirs, self.bin_colormap))
                else:
                    task_pool.add(executor.submit(_dump_img, photo, self.save_dirs))
            for res in tqdm(as_completed(task_pool), desc="saving img result", total=len(task_pool)):
                res.result()

    def take_photo(self, ngp, poses, new_d=None, random_bg=False, appearance_embedding=None, image_names=None, progress_bar=True):
        poses = torch.FloatTensor(poses).view(-1, 3, 4)
        kwargs = {'test_time': True,
                  'appearance_embedding': appearance_embedding,
                  'random_bg': random_bg}
        if ngp.scale > 0.5:
            kwargs['exp_step_factor'] = 1 / 256

        photos = []
        iter_obj = tqdm(enumerate(poses), desc="rendering photos", total=len(poses)) \
                    if progress_bar else enumerate(poses)
        for i, pose in iter_obj:
            rays_o, rays_d = get_rays(self.directions, pose)
            if appearance_embedding is None:
                kwargs['appearance_embedding'] = ngp.get_appearance_embedding()
            if new_d is not None:
                kwargs['new_d'] = torch.FloatTensor(new_d).to(ngp.device)
            #     kwargs['appearance_embedding'] = ngp.get_appearance_embedding(torch.as_tensor(i).to(ngp.device))
            results = render(ngp, rays_o.to(ngp.device), rays_d.to(ngp.device), **kwargs)
            image = rearrange(results['color'].cpu().numpy(), '(h w) c -> h w c', h=self.img_wh[1])
            depth = rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=self.img_wh[1])
            photo = {"pose_id": i, "image": image, "depth": depth}
            if image_names is not None:
                photo["image_name"] = image_names[i]
            photos.append(photo)
        return photos


def project_points(K, xs, ys, depths, M):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    src_cam_coord_pts = depths[..., None] * np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs)], axis=-1) # (N, 3)
    ps_homo = np.concatenate([src_cam_coord_pts, np.ones_like(src_cam_coord_pts[..., :1])], axis=-1) # (N, 4)
    dst_cam_coord_pts = (ps_homo @ M.T)[..., :-1] # (N, 3)
    dst_xs = fx * dst_cam_coord_pts[..., 0] / dst_cam_coord_pts[..., 2] + cx
    dst_ys = fy * dst_cam_coord_pts[..., 1] / dst_cam_coord_pts[..., 2] + cy
    return dst_xs, dst_ys


def match_points_by_points3D_id(src_pts3Did_to_xy: dict, dst_pts3Did_to_xy: dict):
    matched_src_points = []
    matched_dst_points = []
    for pts3Did in set(src_pts3Did_to_xy.keys()).intersection(set(dst_pts3Did_to_xy.keys())):
        src_xy = src_pts3Did_to_xy[pts3Did]
        dst_xy = dst_pts3Did_to_xy[pts3Did]
        if any([i < 0 for xy in [src_xy, dst_xy] for i in xy]):
            continue
        matched_src_points.append(list(src_xy))
        matched_dst_points.append(list(dst_xy))
    return np.asarray(matched_src_points).astype(int), np.asarray(matched_dst_points).astype(int)
    

def bi_project_points(K, src_points, src_img_wh, src_depth_map, dst_img_wh, dst_depth_map, M, sfm_matched_points=None):
    src_matched_points_list = []
    dst_matched_points_list = []
    src_mask = np.zeros(src_img_wh[::-1], dtype=np.uint8)
    for i, points in enumerate(src_points, start=1):
        cv2.fillPoly(src_mask, [points.astype(np.int32)], i)
        src_coords = np.argwhere(src_mask == i)[..., ::-1]
        xs, ys = src_coords[:, 0]+0.5, src_coords[:, 1]+0.5
        src_depths = get_corresponding_depth(src_depth_map, src_coords)
        # 第一次投影：将 src_points 投影到 dst_img 上
        dst_xs, dst_ys = project_points(K, xs, ys, src_depths, M)
        # 第一次检查投影点的 validity - 判断 dst_coords 是否在 dst_img 上
        is_valid = (0 <= dst_xs) & (dst_xs <= dst_img_wh[0]) & (0 <= dst_ys) & (dst_ys <= dst_img_wh[1])
        biproj_success = False
        if is_valid.sum() > 0:
            # 第二次投影：将 dst_points 投影回 src_img 上
            dst_coords = np.stack([dst_xs, dst_ys], axis=-1)
            # 过滤无效点
            src_coords = src_coords[is_valid]
            dst_coords = dst_coords[is_valid]
            dst_depths = get_corresponding_depth(dst_depth_map, dst_coords)
            xs_, ys_ = project_points(K, dst_coords[:, 0], dst_coords[:, 1], dst_depths, np.linalg.inv(M))
            reprojected_coords = np.stack([xs_, ys_], axis=-1)
            # 第二次检查投影点的 validity - 判断 reprojected_coords 是否在 (src_mask == i) 上
            is_valid =src_mask[np.clip(ys_, 0, src_img_wh[1]-1).astype(int), np.clip(xs_, 0, src_img_wh[0]-1).astype(int)] == i
            if is_valid.sum() > 0:
                src_coords = src_coords[is_valid]
                dst_coords = dst_coords[is_valid]
                reprojected_coords = reprojected_coords[is_valid]
                # 按照 双向投影后的误差从小到大排序
                errors = np.linalg.norm(src_coords-reprojected_coords, axis=-1)
                valid_sort_ids = errors < min(np.median(errors), np.mean(errors))  # 取小于中位数和平均数的那一部分
                src_coords = src_coords[valid_sort_ids].astype(np.float32)
                dst_coords = dst_coords[valid_sort_ids].astype(np.float32)
                biproj_success = True

        if sfm_matched_points is not None:
            points_in_region = src_mask[np.clip(sfm_matched_points[0][:, 1], 0, src_img_wh[1]-1), np.clip(sfm_matched_points[0][:, 0], 0, src_img_wh[0]-1)] == i
            if biproj_success:
                src_coords = np.concatenate([src_coords, sfm_matched_points[0][points_in_region]])
                dst_coords = np.concatenate([dst_coords, sfm_matched_points[1][points_in_region]])
            else:
                src_coords = sfm_matched_points[0][points_in_region].astype(np.float32)
                dst_coords = sfm_matched_points[1][points_in_region].astype(np.float32)

        src_matched_points_list += [src_coords]
        dst_matched_points_list += [dst_coords]
        
    return src_matched_points_list, dst_matched_points_list


def draw_anno_mask(annos, img_wh, dilate_kernel_size=15):
    mask = np.zeros(img_wh[::-1], dtype=np.uint8)
    polygons = [np.asarray(anno['points']).astype(int) for anno in annos]
    cv2.fillPoly(mask, polygons, 255)
    if dilate_kernel_size > 0:
        kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask.astype(bool)


def group_and_compute_H(src_matched_points_list, dst_matched_points_list, src_anno_mask=None):
    # src_matched_points_list 包含了 biproject 和 sfm_matched 挑选的两部分，dst_matched_points_list 也是
    if src_anno_mask is None:
        # 认为全部的点都在一个平面上，属于一组
        points_region_ids = [1] * len(src_matched_points_list)
    else:
        # 对mask进行分区域，将每个label分配到各的区域
        labeled_mask = measure.label(src_anno_mask.astype(bool))
        points_region_ids = []
        for src_matched_points in src_matched_points_list:
            if len(src_matched_points) == 0:
                points_region_ids.append(None)
                continue
            ref_point = src_matched_points.mean(axis=0).astype(int)
            region_id = labeled_mask[ref_point[1], ref_point[0]]
            points_region_ids.append(region_id)
        
    
    region_Hs = {}
    for region_id in set(points_region_ids):
        if region_id is None:
            continue
        src_points = np.concatenate([src_matched_points_list[i] for i, rid in enumerate(points_region_ids) if rid == region_id])
        dst_points = np.concatenate([dst_matched_points_list[i] for i, rid in enumerate(points_region_ids) if rid == region_id])
        try:
            transformation_matrix, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 2.5)  # 计算透视变换矩阵
        except:
            print("src_points: ", src_points)
            print("dst_points: ", dst_points)
            # input("pause")
            transformation_matrix = None
        region_Hs[region_id] = transformation_matrix

    region_id_cnt = Counter(filter(lambda x: x is not None, points_region_ids))
    if len(region_id_cnt) == 0:
        return defaultdict(int), defaultdict(lambda: None)
    max_region_id = max(region_id_cnt, key=lambda x: region_id_cnt[x])
    for i in range(len(points_region_ids)):
        if points_region_ids[i] is None:
            points_region_ids[i] = max_region_id
            
    return points_region_ids, region_Hs  # 使用时遍历 points_region_ids, 根据 region_id 获取对应的 region_Hs 然后对 src_matched_points_list 所对应的 points 进行投影


def transmit_annos(K, annos, src_img_wh, src_depth_map, dst_img_wh, dst_depth_map, M, sfm_matched_points=None, all_txt_on_same_plane=True):
    src_points_list = [np.asarray(anno["points"]) for anno in annos]
    src_matched_points_list, dst_matched_points_list = bi_project_points(K, src_points_list, src_img_wh, src_depth_map, dst_img_wh, dst_depth_map, M, sfm_matched_points=sfm_matched_points)
    dilate_kernel_size = -1 if all_txt_on_same_plane else 25
    src_anno_mask = draw_anno_mask(annos, src_img_wh, dilate_kernel_size)
    points_region_ids, region_Hs = group_and_compute_H(src_matched_points_list, dst_matched_points_list, src_anno_mask=src_anno_mask)

    dst_annos = []
    for i in range(len(annos)):
        uncertain = False
        src_points = np.asarray(annos[i]['points'])
        transcription = annos[i]['transcription']
        points_region_id = points_region_ids[i]
        transformation_matrix = region_Hs[points_region_id]
        if transformation_matrix is not None:   # 如果 存在 transformation_matrix 则进行投影
            src_points_homo = np.concatenate([src_points, np.ones_like(src_points[..., :1])], axis=-1).astype(np.float32)
            dst_points_homo = src_points_homo @ transformation_matrix.T  # 将 src_img 上的 bbox 顶点 points 坐标映射到 dst_img 上
            dst_point_xs, dst_point_ys = dst_points_homo[..., 0]/dst_points_homo[..., 2], dst_points_homo[..., 1]/dst_points_homo[..., 2]
        else:  # 否则 退化到直接使用坐标点投影
            src_points_depths = get_corresponding_depth(src_depth_map, src_points)
            dst_point_xs, dst_point_ys = project_points(K, src_points[..., 0], src_points[..., 1], src_points_depths, M)
            uncertain = True

        if np.all(dst_point_xs < 0) or np.all(dst_point_ys < 0) or np.all(dst_point_xs >= dst_img_wh[0]) or np.all(dst_point_ys >= dst_img_wh[1]):
            continue # dst_points 不在 dst_img 上
        else:
            dst_point_xys = np.stack([dst_point_xs, dst_point_ys], axis=-1).astype(int).tolist()
            anno = {"transcription": transcription, "points": dst_point_xys}
            if uncertain:
                anno["uncertain"] = uncertain
            dst_annos.append(anno)

    return dst_annos


class TextNeRFEngine(Camera):
    def __init__(self, ngp, dataset, all_txt_on_same_plane, save_dir="outputs", vis_depth=True, use_euclidean_distance=False):
        self.ngp = ngp
        self.dataset = dataset
        self.K = self.dataset.K
        self.img_wh = self.dataset.img_wh
        self.all_txt_on_same_plane = all_txt_on_same_plane
        self.text_3ds = [] # 场景中 文本的 3D 属性
        self.custom_text_images = dict()
        super().__init__(self.K, self.img_wh, self.dataset.color_mode, save_dir, vis_depth)
        
        if use_euclidean_distance:
            self.sim_kernel = lambda pose1, pose2: -np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])
        else:
            def pose_sim_kernel(pose_i, pose_j):
                quaternion_i, quaternion_j = R_to_quaternion(pose_i[:3, :3]), R_to_quaternion(pose_j[:3, :3])
                return np.dot(quaternion_i, quaternion_j) / (np.linalg.norm(quaternion_i) * np.linalg.norm(quaternion_j))
            self.sim_kernel = pose_sim_kernel
        self.pose_similarity_matrix = self.compute_pose_similarity() 
        self.M = self.get_homography_matrix() 
        self.depth_cache = dict()
        self.annotations = dict()  
        self.uncertain_anno_pose_ids = []
        
    def get_homography_matrix(self):
        # 将 pose_i 坐标系下的点 -> pose_j 坐标系的变换矩阵
        num_poses = len(self.dataset.poses)
        M_table = np.zeros((num_poses, num_poses, 4, 4))
        bottom = np.array([[0, 0, 0, 1.]])
        for i, pose_i in enumerate(self.dataset.poses):
            pose_i = np.concatenate([pose_i, bottom], axis=0)
            for j, pose_j in enumerate(self.dataset.poses):
                pose_j = np.concatenate([pose_j, bottom], axis=0)
                M_table[i, j] = np.linalg.inv(pose_j) @ pose_i
        return M_table    
        
    def compute_pose_similarity(self):
        poses = self.dataset.poses
        similarity_matrix = np.zeros((len(poses), len(poses)))
        for i in range(len(poses)):
            similarity_matrix[i, i] = 1
            for j in range(i+1, len(poses)):
                similarity_matrix[i, j] = similarity_matrix[j, i] = self.sim_kernel(poses[i], poses[j])
        
        return similarity_matrix
        
    def get_anno_by_pose_id(self, pose_id, cache_ids=[]):
        if self.annotations.get(pose_id, None) is not None:
            annos = self.annotations[pose_id]
        else:
            # 寻找最近邻 pose 的 id
            pose_similarities = self.pose_similarity_matrix[pose_id].copy()
            if self.uncertain_anno_pose_ids:
                pose_similarities[self.uncertain_anno_pose_ids] = -10000
            if cache_ids:
                pose_similarities[cache_ids] = -10000
            ref_pose_id = np.argmax(pose_similarities)
            ref_pose_annos = self.get_anno_by_pose_id(ref_pose_id, cache_ids+[ref_pose_id])
            if ref_pose_id in self.uncertain_anno_pose_ids:
                annos = self.get_anno_by_pose_id(pose_id, cache_ids)
            else:
                if self.depth_cache.get(ref_pose_id, None) is None:
                    ref_depth_map = self.take_photo(self.ngp, self.dataset.poses[ref_pose_id], progress_bar=False)[0]["depth"]
                    self.depth_cache[ref_pose_id] = ref_depth_map
                else:
                    ref_depth_map = self.depth_cache[ref_pose_id]
                    
                if self.depth_cache.get(pose_id, None) is None:
                    cur_depth_map = self.take_photo(self.ngp, self.dataset.poses[pose_id], progress_bar=False)[0]["depth"]
                    self.depth_cache[pose_id] = cur_depth_map
                else:
                    cur_depth_map = self.depth_cache[pose_id]

                M = self.M[ref_pose_id, pose_id]
                sfm_matched_points = match_points_by_points3D_id(self.dataset.pts3D_id_to_xys[ref_pose_id], self.dataset.pts3D_id_to_xys[pose_id])
                annos = transmit_annos(self.K, ref_pose_annos, self.img_wh, ref_depth_map, self.img_wh, cur_depth_map, M, 
                                    sfm_matched_points=sfm_matched_points, all_txt_on_same_plane=self.all_txt_on_same_plane)
                
                if any(ann.get("uncertain", False) for ann in annos) and pose_id not in self.uncertain_anno_pose_ids:
                    self.uncertain_anno_pose_ids.append(pose_id)

                annos = get_text_pose(annos, cur_depth_map, self.K, use_mean_points_as_t=True, compute_vis_scale=False)

                self.annotations[pose_id] = annos
            
        return annos
        
    def get_anno_by_pose(self, pose):
        # 先匹配到最近的 dataset 中的 pose
        matched_data_pose_id = None
        matched_similarity = None
        for i in range(len(self.dataset.poses)):
            per_similarity = self.sim_kernel(pose, self.dataset.poses[i])
            if (matched_data_pose_id is None) or (per_similarity > matched_similarity):
                matched_data_pose_id = i
                matched_similarity = per_similarity
        
        # 将匹配到的 data_pose 的 annotation 投影到所给 pose 视角下
        ref_pose_annos = self.get_anno_by_pose_id(matched_data_pose_id)
        if self.depth_cache.get(matched_data_pose_id, None) is None:
            ref_depth_map = self.take_photo(self.ngp, self.dataset.poses[matched_data_pose_id], progress_bar=False)[0]["depth"]
            self.depth_cache[matched_data_pose_id] = ref_depth_map
        else:
            ref_depth_map = self.depth_cache[matched_data_pose_id]
        # 利用 ref_pose 的 depth 信息将其 annotation 投影到 src_pose 下
        bottom = np.array([[0, 0, 0, 1.]])
        pose_i = np.concatenate([self.dataset.poses[matched_data_pose_id], bottom], axis=0)
        pose_j = np.concatenate([pose, bottom], axis=0)
        M = np.linalg.inv(pose_j) @ pose_i
        w, h = self.img_wh
        annos = []
        for ref_anno_item in ref_pose_annos:  # ref_anno_item 是一个 dict
            ref_points = np.asarray(ref_anno_item["points"])
            ref_xs, ref_ys = ref_points[:, 0], ref_points[:, 1]
            ref_point_depths = get_corresponding_depth(ref_depth_map, ref_points)
            xs, ys = project_points(self.K, ref_xs, ref_ys, ref_point_depths, M)
            if all(x < 0 or x >= w for x in xs) and all(y < 0 or y >= h for y in ys):
                continue  # 在当前 pose 下不能看见该 text，跳过
            points = np.stack([xs, ys], axis=-1).astype(int).tolist()
            annos.append({"transcription": ref_anno_item["transcription"], "points": points})
            
        cur_depth_map = self.take_photo(self.ngp, pose, progress_bar=False)[0]["depth"]
        cur_annos = get_text_pose(annos, cur_depth_map, self.K, use_mean_points_as_t=True, compute_vis_scale=False)
        return cur_annos
        
    def annotate_dataset(self, hand_annotation_path, annotation_save_path):
        # 提前将 人工标注的信息 load 到 self.annotations，否则会死循环
        img_names = list(map(lambda path: os.path.splitext(os.path.basename(path))[0], self.dataset.img_paths))
        with open(hand_annotation_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    image_path, annos_str = line.split("\t")
                    image_name = os.path.splitext(os.path.basename(image_path))[0]
                    try:
                        pose_id = img_names.index(image_name)
                    except ValueError as e:
                        print(e)
                        continue
                    annos = json.loads(annos_str)
                    if self.depth_cache.get(pose_id, None) is None:
                        cur_depth_map = self.take_photo(self.ngp, self.dataset.poses[pose_id], progress_bar=False)[0]["depth"]
                        self.depth_cache[pose_id] = cur_depth_map
                    else:
                        cur_depth_map = self.depth_cache[pose_id]
                    annos = get_text_pose(annos, cur_depth_map, self.K, use_mean_points_as_t=True, compute_vis_scale=False)
                    self.annotations[pose_id] = annos
        if len(self.annotations) == 0:
            print("No valid annotation!")
            print("All images: ", img_names)
            return

        with open(annotation_save_path, "w", encoding='utf-8') as f:
            for i, image_path in tqdm(enumerate(self.dataset.img_paths), total=len(self.dataset.img_paths)):
                image_path = os.path.basename(image_path)
                annos = json.dumps(self.get_anno_by_pose_id(i), ensure_ascii=False)
                f.write(image_path + "\t" + annos + "\n")
        print(f"All photos have been annotated, and the result is saved to {annotation_save_path}")
                
    def annotate(self, pose, K=None, img_wh=None):
        if K is None:
            K = self.K
        if img_wh is None:
            img_wh = self.img_wh
        bottom = np.array([[0, 0, 0, 1.]])
        c2w_4x4 = np.concatenate([pose, bottom])
        w2c_4x4 = np.linalg.inv(c2w_4x4)
        w, h = img_wh

        annos = []
        for text_label in sorted(self.text_3ds, key=lambda x: x["text_id"]):
            # project points_world_3d to pixel points 
            points_world_3d = text_label["points_world_3d"]
            points_cam_3d_homo = np.concatenate([points_world_3d, np.ones_like(points_world_3d[:, :1])], axis=-1) @ w2c_4x4.T
            points_cam_3d = points_cam_3d_homo[:, :3]
            points_homo = points_cam_3d @ K.T
            points = (points_homo[:, :2] / points_homo[:, 2:3])
            points[:, 0] = np.clip(points[:, 0], 0, w-1)
            points[:, 1] = np.clip(points[:, 1], 0, h-1)
            points = points.astype(np.int32)
            
            # get transcription
            transcription = text_label["transcription"]
            
            # project text_pose_world to camera coordinate
            text_pose_world = text_label["text_pose_world"]  # [3, 4]
            text_pose_world_4x4 = np.concatenate([text_pose_world, bottom])
            text_pose_cam = (w2c_4x4 @ text_pose_world_4x4)[:3, :]
            
            # project text_area_points to pixel points, but not constrained in (w, h)
            area_points_world_3d = text_label["area_points_world_3d"]
            area_points_cam_3d_homo = np.concatenate([area_points_world_3d, np.ones_like(area_points_world_3d[:, :1])], axis=-1) @ w2c_4x4.T
            area_points_cam_3d = area_points_cam_3d_homo[:, :3]
            area_points_homo = area_points_cam_3d @ K.T
            area_points = (area_points_homo[:, :2] / area_points_homo[:, 2:3])
            
            annos.append({"transcription": transcription,
                          "points": points.tolist(),
                          "text_pose": text_pose_cam.tolist(),
                          "area_points": area_points.tolist()})
        return annos
   