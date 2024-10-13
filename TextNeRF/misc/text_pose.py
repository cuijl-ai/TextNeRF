import numpy as np
import math
import cv2

from PIL import Image

import sys
sys.path.append("..")
from datasets.utils.ray_utils import normalize
from .depth_utils import get_corresponding_depth


def R_to_quaternion(R) -> np.ndarray:
    """Convert rotation matrix to quaternion"""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    # symmetric matrix K
    K = np.array(
        [
            [m00 - m11 - m22, 0.0, 0.0, 0.0],
            [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
            [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0
    # quaternion is eigenvector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    q = V[np.array([3, 0, 1, 2]), np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def quaternion_to_R(quaternion) -> np.ndarray:
    """Return rotation matrix from quaternion.
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [1.0 - q[2, 2] - q[3, 3],       q[1, 2] - q[3, 0],        q[1, 3] + q[2, 0]],
            [q[1, 2] + q[3, 0],       1.0 - q[1, 1] - q[3, 3],        q[2, 3] - q[1, 0]],
            [q[1, 3] - q[2, 0],             q[2, 3] + q[1, 0],  1.0 - q[1, 1] - q[2, 2]],
        ]
    )


def calculate_intersection(point1, point2, point3, point4):
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    x4, y4 = point4

    # 计算直线方程的系数
    a1 = y3 - y1
    b1 = x1 - x3
    c1 = x3 * y1 - x1 * y3

    a2 = y4 - y2
    b2 = x2 - x4
    c2 = x4 * y2 - x2 * y4

    # 计算交点坐标
    determinant = a1 * b2 - a2 * b1
    if determinant == 0:
        return None
    else:
        x = (c1 * b2 - c2 * b1) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        return [x, y]


def get_cam_3d_points(points_2d, K, depth):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xs, ys = points_2d[..., 0], points_2d[..., 1]
    cam_pts_3d = depth * np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs)], axis=-1)  # (N, 3)
    return cam_pts_3d


def proj_cam_3d_pts_to_img(cam_pts_3d, K):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    points_2d = np.array([[fx, fy]]) * (cam_pts_3d[:, :2] / cam_pts_3d[:, 2:3]) + np.array([[cx, cy]])
    return points_2d


def get_text_box_world_3d(text_labels, depth_map, K, c2w_4x4):
    text_points_world_3d = []
    for text_ins in text_labels:
        # points_2d = np.array(sort_points(text_ins["points"])).astype(np.int32)  
        points_2d = np.array(text_ins["points"]).astype(np.int32)  
        depths = get_corresponding_depth(depth_map, points_2d)[:, None]
        points_cam_3d = get_cam_3d_points(points_2d, K, depths)  
        points_cam_3d_homo = np.concatenate([points_cam_3d, np.ones_like(points_cam_3d[:, :1])], axis=-1)
        points_world_3d = (points_cam_3d_homo @ c2w_4x4.T)[:, :3]  # [4, 3]
        text_points_world_3d += [points_world_3d]
    return text_points_world_3d


def get_text_pose_world(points_world_3d, c2w_4x4=False):
    p1, p2, p3, p4 = points_world_3d[0], points_world_3d[1], points_world_3d[2], points_world_3d[3]
    axis_z = normalize(np.cross(p3-p1, p2-p4))
    axis_x = normalize(normalize(p2-p1) + normalize(p3-p4))
    axis_y = normalize(np.cross(axis_z, axis_x))
    translation = points_world_3d.mean(axis=0)  
    text_pose_world = np.stack([axis_x, axis_y, axis_z, translation]).T  # [3, 4]
    if c2w_4x4:
        text_pose_world = np.concatenate([text_pose_world, np.array([[0, 0, 0, 1.0]])])
    return text_pose_world  


def get_mean_text_points_world_3d(text_points_world_3d):
    mean_text_points_world_3d = {}
    for text_id, points_world_3d_list in text_points_world_3d.items():
        points_world_3d_list = list(map(lambda x: x["points_world_3d"], points_world_3d_list))
        points_world_3d = np.stack(points_world_3d_list, axis=1).mean(axis=1)  # [4, 3]
        text_pose_world = get_text_pose_world(points_world_3d)
        mean_text_points_world_3d[text_id] = {"points_world_3d": points_world_3d,
                                              "text_pose_world": text_pose_world}
    return mean_text_points_world_3d


def get_median_text_points_world_3d(text_points_world_3d):
    median_text_points_world_3d = {}
    for text_id, points_world_3d_list in text_points_world_3d.items():
        points_world_3d_list = list(map(lambda x: x["points_world_3d"], points_world_3d_list))
        points_world_3d = np.median(np.stack(points_world_3d_list, axis=1), axis=1)  # [4, 3]
        text_pose_world = get_text_pose_world(points_world_3d)
        median_text_points_world_3d[text_id] = {"points_world_3d": points_world_3d,
                                                "text_pose_world": text_pose_world}
    return median_text_points_world_3d


def get_max_text_points_world_3d(text_points_world_3d):
    max_text_points_world_3d = {}
    for text_id, points_world_3d_list in text_points_world_3d.items():
        points_world_3d_list = list(map(lambda x: x["points_world_3d"], points_world_3d_list))
        center_point_3d = np.mean(np.concatenate(points_world_3d_list, axis=0), axis=0)  # [3,]
        vertices = np.stack(points_world_3d_list, axis=1)  # [4, N, 3]
        distance = np.linalg.norm(center_point_3d - vertices, axis=-1)  # [4, N]
        points_idx = np.argmax(distance, axis=-1)  # [4,]
        points_world_3d = vertices[np.arange(vertices.shape[0]), points_idx]  # [4, 3]
        text_pose_world = get_text_pose_world(points_world_3d)
        max_text_points_world_3d[text_id] = {"points_world_3d": points_world_3d,
                                             "text_pose_world": text_pose_world}
    return max_text_points_world_3d


def get_text_points_world_3d_allocate(text_points_world_3d, poses, pose_sim_kernel=None):
    poses_cache = {}
    points_world_3d_cache = {}
    for text_id, points_world_3d_list in text_points_world_3d.items():
        for points_dict in points_world_3d_list:
            pose_id = points_dict["pose_id"]
            points_world_3d = points_dict["points_world_3d"]
            text_pose_world = get_text_pose_world(points_world_3d)
            if pose_id not in poses_cache:
                poses_cache[pose_id] = poses[pose_id]
            if pose_id not in points_world_3d_cache:
                points_world_3d_cache[pose_id] = {}
            points_world_3d_cache[pose_id].update({text_id: {"points_world_3d": points_world_3d,
                                                             "text_pose_world": text_pose_world}})
    # check if some text are omitted in some poses
    median_text_points_world_3d = get_median_text_points_world_3d(text_points_world_3d)
    for pose_id in points_world_3d_cache:
        not_contained_text = {text_id: median_text_points_world_3d[text_id]
                              for text_id in points_world_3d_cache[pose_id] if text_id not in text_points_world_3d}
        if len(not_contained_text) == 0:
            continue
        points_world_3d_cache[pose_id].update(not_contained_text)

    if pose_sim_kernel is None:
        def pose_sim_kernel(pose_i, pose_j):
            quaternion_i, quaternion_j = R_to_quaternion(pose_i[:3, :3]), R_to_quaternion(pose_j[:3, :3])
            return np.dot(quaternion_i, quaternion_j) / (np.linalg.norm(quaternion_i) * np.linalg.norm(quaternion_j))

    def dynamic_text_points_world_3d_func(pose):
        get_near_pose_id = max(list(poses_cache.items()),
                               key=lambda id_pose_pair: pose_sim_kernel(pose, id_pose_pair[1]))[0]
        return points_world_3d_cache[get_near_pose_id]

    return dynamic_text_points_world_3d_func


def get_text_pose(text_labels, depth_map, K, use_mean_points_as_t=True, compute_vis_scale=False):
    all_points_3d = []
    for text_ins in text_labels:
        if use_mean_points_as_t:
            points_2d = np.array(text_ins["points"]).astype(np.int32)
            
        else: # use intersection
            p1, p2, p3, p4 = text_ins["points"]  
            p5 = calculate_intersection(p1, p2, p3, p4)  
            points_2d = np.array([p1, p2, p3, p4, p5]).astype(np.int32)
        
        depths = get_corresponding_depth(depth_map, points_2d)[:, None]
        points_3d = get_cam_3d_points(points_2d, K, depths)  # [4, 3] or [5, 3]
        all_points_3d.append(points_3d[:4, :])
        # 在相机坐标系空间中计算文本区域的位姿
        axis_z = normalize(np.cross(points_3d[2] - points_3d[0], points_3d[1] - points_3d[3]))
        axis_x = normalize((points_3d[2] - points_3d[0])+(points_3d[1] - points_3d[3]))
        axis_y = normalize(np.cross(axis_x, -axis_z))
        if use_mean_points_as_t:
            translation = points_3d.mean(axis=0)
        else:
            translation = points_3d[4]
        
        text_pose = np.stack([axis_x, axis_y, axis_z, translation]).T.tolist()  # [3, 4]
        text_ins["text_pose"] = text_pose
    if not compute_vis_scale:
        return text_labels
    else:
        all_points_3d = np.stack(all_points_3d)  # [N, 4, 3]
        l1 = np.linalg.norm(all_points_3d[:, 1, :]-all_points_3d[:, 0, :], axis=-1)
        l2 = np.linalg.norm(all_points_3d[:, 2, :]-all_points_3d[:, 1, :], axis=-1)
        l3 = np.linalg.norm(all_points_3d[:, 3, :]-all_points_3d[:, 2, :], axis=-1)
        l4 = np.linalg.norm(all_points_3d[:, 0, :]-all_points_3d[:, 3, :], axis=-1)
        arr_scale = np.median(np.concatenate([l1, l2, l3, l4]))
        return text_labels, arr_scale


def visualize_text_pose(text_labels, K, pil_image, pose_only=False, scale=1.0):
    # PIL in, PIL out
    image = np.asarray(pil_image)
    if pose_only:
        image = np.full_like(image, fill_value=255, dtype=np.uint8)

    # draw text box
    polygon_color = (0, 0, 0)
    polygon_points = [np.array(text_ins["points"]).reshape((-1, 1, 2)).astype(np.int32) for text_ins in text_labels]
    cv2.polylines(image, polygon_points, isClosed=True, color=polygon_color, thickness=3)

    # project text pose to image plane
    pts_need_to_project = []
    for text_ins in text_labels:
        text_pose = np.array(text_ins["text_pose"])
        o = text_pose[:, -1]
        ox = o + text_pose[:, 0] * scale
        oy = o + text_pose[:, 1] * scale
        oz = o + text_pose[:, 2] * scale
        pts_need_to_project += [o, ox, oy, oz]
    pts_need_to_project = np.stack(pts_need_to_project)
    projected_pts_2d = proj_cam_3d_pts_to_img(pts_need_to_project, K).reshape(-1, 4, 2).astype(np.int32)

    # draw projected pose to image
    color_line1, color_line2, color_line3 = (255, 0, 0), (0, 255, 0), (0, 0, 255)
    for pts_2d in projected_pts_2d:
        o, x, y, z = pts_2d[0], pts_2d[1], pts_2d[2], pts_2d[3]
        arrowed_line1, arrowed_line2, arrowed_line3 = [o, x], [o, y], [o, z]
        cv2.arrowedLine(image, arrowed_line1[0], arrowed_line1[1], color=color_line1, thickness=2, tipLength=0.1)
        cv2.arrowedLine(image, arrowed_line2[0], arrowed_line2[1], color=color_line2, thickness=2, tipLength=0.1)
        cv2.arrowedLine(image, arrowed_line3[0], arrowed_line3[1], color=color_line3, thickness=2, tipLength=0.1)

    return Image.fromarray(image)


def visualize_text_anno(text_labels, pil_image):
    # PIL in, PIL out
    image = np.asarray(pil_image)
    # draw text box
    polygon_color = (0, 0, 0)
    polygon_points = [np.array(text_ins["points"]).reshape((-1, 1, 2)).astype(np.int32) for text_ins in text_labels]
    cv2.polylines(image, polygon_points, isClosed=True, color=polygon_color, thickness=3)
    return Image.fromarray(image)
