import numpy as np
import re
import cv2
import sys


def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


def get_corresponding_depth(depth_map, points_2d):
    h, w = depth_map.shape
    i, j = points_2d[..., 1], points_2d[..., 0]
    return depth_map[np.clip(i, 0, h-1).astype(int), np.clip(j, 0, w-1).astype(int)]


def interpolation_depth(depth_map, points_2d, interpolation_mode="bilinear"):
    """
    Input:
        - depth_map: 深度图 with shape of [h, w]
        - points_2d: 坐标点
    Output:
        - points_2d对应的深度值（使用给定的interpolation_mode进行插值）
    """
    h, w = depth_map.shape
    # 目前仅实现双线性插值的版本
    if interpolation_mode == "bilinear":
        # 1. get the 4 neighbors of each given point
        ref_point = points_2d.astype(np.int32).astype(np.float32)+0.5  # 认为像素值在像素中点
        mask_lt = (ref_point[..., 0] > points_2d[..., 0]) & (ref_point[..., 1] > points_2d[..., 1])
        mask_rt = (ref_point[..., 0] < points_2d[..., 0]) & (ref_point[..., 1] > points_2d[..., 1])
        mask_rb = (ref_point[..., 0] < points_2d[..., 0]) & (ref_point[..., 1] < points_2d[..., 1])
        mask_lb = (ref_point[..., 0] > points_2d[..., 0]) & (ref_point[..., 1] < points_2d[..., 1])

        x1y1 = np.zeros_like(points_2d)
        x2y2 = np.zeros_like(points_2d)
        x3y3 = np.zeros_like(points_2d)
        x4y4 = np.zeros_like(points_2d)

        if mask_lt.any():
            x1y1[mask_lt][..., 0], x1y1[mask_lt][..., 1] = ref_point[mask_lt][..., 0] - 1.0, ref_point[mask_lt][..., 1] - 1.0
            # x2y2[mask_lt][..., 0], x2y2[mask_lt][..., 1] = ref_point[mask_lt][..., 0], ref_point[mask_lt][..., 1] - 1.0
            x3y3[mask_lt][..., 0], x3y3[mask_lt][..., 1] = ref_point[mask_lt][..., 0], ref_point[mask_lt][..., 1]
            # x4y4[mask_lt][..., 0], x4y4[mask_lt][..., 1] = ref_point[mask_lt][..., 0] - 1.0, ref_point[mask_lt][..., 1]
        if mask_rt.any():
            x1y1[mask_rt][..., 0], x1y1[mask_rt][..., 1] = ref_point[mask_rt][..., 0], ref_point[mask_rt][..., 1] - 1.0
            # x2y2[mask_rt][..., 0], x2y2[mask_rt][..., 1] = ref_point[mask_rt][..., 0] + 1.0, ref_point[mask_rt][..., 1] - 1.0
            x3y3[mask_rt][..., 0], x3y3[mask_rt][..., 1] = ref_point[mask_rt][..., 0] + 1.0, ref_point[mask_rt][..., 1]
            # x4y4[mask_rt][..., 0], x4y4[mask_rt][..., 1] = ref_point[mask_rt][..., 0], ref_point[mask_rt][..., 1]
        if mask_rb.any():
            x1y1[mask_rb][..., 0], x1y1[mask_rb][..., 1] = ref_point[mask_rb][..., 0], ref_point[mask_rb][..., 1]
            # x2y2[mask_rb][..., 0], x2y2[mask_rb][..., 1] = ref_point[mask_rb][..., 0] + 1.0, ref_point[mask_rb][..., 1]
            x3y3[mask_rb][..., 0], x3y3[mask_rb][..., 1] = ref_point[mask_rb][..., 0] + 1.0, ref_point[mask_rb][..., 1] + 1.0
            # x4y4[mask_rb][..., 0], x4y4[mask_rb][..., 1] = ref_point[mask_rb][..., 0], ref_point[mask_rb][..., 1] + 1.0
        if mask_lb.any():
            x1y1[mask_lb][..., 0], x1y1[mask_lb][..., 1] = ref_point[mask_lb][..., 0] - 1.0, ref_point[mask_lb][..., 1]
            # x2y2[mask_lb][..., 0], x2y2[mask_lb][..., 1] = ref_point[mask_lb][..., 0], ref_point[mask_lb][..., 1]
            x3y3[mask_lb][..., 0], x3y3[mask_lb][..., 1] = ref_point[mask_lb][..., 0], ref_point[mask_lb][..., 1] + 1.0
            # x4y4[mask_lb][..., 0], x4y4[mask_lb][..., 1] = ref_point[mask_lb][..., 0] - 1.0, ref_point[mask_lb][..., 1] + 1.0

        # 2. 计算四个像素的权重
        x, y = points_2d[..., 0], points_2d[..., 1]
        x1, y1 = x1y1[..., 0], x1y1[..., 1]
        x2, y2 = x3y3[..., 0], x3y3[..., 1]
        w1 = (x2 - x) * (y2 - y)
        w2 = (x - x1) * (y2 - y)
        w3 = (x2 - x) * (y - y1)
        w4 = (x - x1) * (y - y1)

        inv_depth_map = 1 / (depth_map + 1e-8)

        def get_inverse_depth(inv_depth_map, i, j):
            return inv_depth_map[np.clip(i, 0, h-1), np.clip(j, 0, w-1)]
        # 使用权重对四个inverse_depth值进行加权平均
        interpolated_inv_depth = (
            w1 * get_inverse_depth(inv_depth_map, y1.astype(np.int32), x1.astype(np.int32)) +\
            w2 * get_inverse_depth(inv_depth_map, y1.astype(np.int32), x2.astype(np.int32)) +\
            w3 * get_inverse_depth(inv_depth_map, y2.astype(np.int32), x1.astype(np.int32)) +\
            w4 * get_inverse_depth(inv_depth_map, y2.astype(np.int32), x2.astype(np.int32))
        )
        interpolated_depth = 1 / interpolated_inv_depth
        return interpolated_depth

    else:
        raise NotImplementedError("Now only support the 'bilinear' interpolation mode...")


def read_pfm(path):
    """Read pfm file.

    Args:
        path (str): path to file

    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale


def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()