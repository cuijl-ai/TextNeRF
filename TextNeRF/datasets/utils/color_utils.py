import cv2
from einops import rearrange
from PIL import Image
import numpy as np


def srgb_to_linear(img):
    limit = 0.04045
    return np.where(img>limit, ((img+0.055)/1.055)**2.4, img/12.92)


def linear_to_srgb(img):
    limit = 0.0031308
    img = np.where(img>limit, 1.055*img**(1/2.4)-0.055, 12.92*img)
    img[img>1] = 1 # "clamp" tonemapper
    return img


def read_image(img_path, img_wh=None, color_mode="RGB", flatten=True):
    img = np.asarray(Image.open(img_path).convert(color_mode), dtype=np.float32)
    if color_mode == "LAB":
        img[:, :, 0] = img[:, :, 0] / 100.0  # L通道归一化到0~1
        img[:, :, 1:] = (img[:, :, 1:] + [128, 128]) / [255, 255]  # a和b通道归一化到0~1
    elif color_mode == "RGB":
        img = img / 255.0
    if img_wh is not None:
        img = cv2.resize(img, img_wh)
    if flatten:
        img = rearrange(img, 'h w c -> (h w) c')
    return img


def read_appearance_image(img_path, img_wh, color_mode="RGB"):
    img = np.asarray(Image.open(img_path).convert(color_mode), dtype=np.float32)
    if color_mode == "LAB":
        img[:, :, 0] = img[:, :, 0] / 100.0  # L通道归一化到0~1
        img[:, :, 1:] = (img[:, :, 1:] + [128, 128]) / [255, 255]  # a和b通道归一化到0~1
    elif color_mode == "RGB":
        img = img / 255.0
    img = cv2.resize(img, img_wh)
    img = rearrange(img, 'h w c -> c h w')
    return img


def read_sem_map(sem_path, img_wh=None, flatten=True):
    sem_map = np.asarray(Image.open(sem_path).convert("P"))
    if img_wh is not None:
        sem_map = cv2.resize(sem_map, img_wh)  # (h, w)
    if flatten:
        return sem_map.reshape(-1)
    else:
        return sem_map
