import os
from tqdm import tqdm
from ..utils.ray_utils import *
from ..utils.color_utils import read_image
from ..utils.colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary


class TextNeRFGeometryDataset(object):
    """ Geometry dataset 以 ray 为 sample 粒度 """
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample
        self.sample_ratio = kwargs.get("sample_ratio", 0.5)
        self.color_mode = kwargs.get("color_mode", "RGB")

        self.read_intrinsics()
        self.json_dir = os.path.join(root_dir, 'labels')
        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)

        if split == 'train':
            self.batch_size = kwargs.get("batch_size", self.img_wh[0] * self.img_wh[1])
            self.ray_sampling_strategy = kwargs.get('ray_sampling_strategy', 'all_images')
        else:
            self.ray_sampling_strategy = 'same_image'

    def read_intrinsics(self):
        """ 主要是用于确认相机内参和 相机坐标系下的 ray direction """
        # Step 1: read and scale intrinsics (same for all images)
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        h = int(camdata[1].height*self.downsample)
        w = int(camdata[1].width*self.downsample)
        self.img_wh = (w, h)

        if camdata[1].model == 'SIMPLE_RADIAL':
            fx = fy = camdata[1].params[0]*self.downsample
            cx = camdata[1].params[1]*self.downsample
            cy = camdata[1].params[2]*self.downsample
        elif camdata[1].model in ['PINHOLE', 'OPENCV']:
            fx = camdata[1].params[0]*self.downsample
            fy = camdata[1].params[1]*self.downsample
            cx = camdata[1].params[2]*self.downsample
            cy = camdata[1].params[3]*self.downsample
        else:
            raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
        self.K = torch.FloatTensor([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0,  0,  1]])
        self.directions = get_ray_directions(h, w, self.K)

    def get_text_label(self, image_name):
        # 判断当前图片是否有对应的文本区域语义标注
        # [{"points": [[], [], ..], "label": "xxx", "group_id": 1}, {...}, ...]
        if os.path.basename(image_name) in self.image_labels:
            return self.image_labels[image_name]
        else:
            return None

    def read_meta(self, split, **kwargs):
        """ 整理好所有需要用于训练的样本集合，这里并没有一下子整理出所欲的光线，而是整理出了所有ray的gt，和pose """
        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
        img_names = [imdata[k].name for k in imdata]
        perm = np.argsort(img_names)

        # read successfully reconstructed images and ignore others
        self.img_paths = [os.path.join(self.root_dir, 'images', name) for name in sorted(img_names)]
        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat(); t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]

        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[perm, :3] # (N_images, 3, 4) cam2world matrices

        pts3d_data = read_points3d_binary(os.path.join(self.root_dir, 'sparse/0/points3D.bin'))
        pts3d = np.array([pts3d_data[k].xyz for k in pts3d_data]) # (N, 3)
        self.poses, self.pts3d = center_poses(poses, pts3d)
        scale = np.linalg.norm(self.poses[..., 3], axis=-1).min()
        self.poses[..., 3] /= scale
        self.pts3d /= scale

        # use every 8th image as test set
        if split == 'train':
            self.img_paths = [x for i, x in enumerate(self.img_paths) if i%8!=0]
            self.poses = np.array([x for i, x in enumerate(self.poses) if i%8!=0])
        elif split == 'val':
            self.img_paths = [x for i, x in enumerate(self.img_paths) if i%8==0]
            self.poses = np.array([x for i, x in enumerate(self.poses) if i%8==0])

        # read color
        self.rays = []
        print(f'Loading {len(self.img_paths)} {split} images ...')
        for img_path in tqdm(self.img_paths):
            buf = [] # buffer for ray attributes: color, etc

            img = read_image(img_path, self.img_wh, color_mode=self.color_mode)
            img = torch.FloatTensor(img)
            buf += [img]

            self.rays += [torch.cat(buf, 1)]

        self.rays = torch.stack(self.rays) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)

    def __len__(self):
        if self.split == "train":
            N_images, hw, _ = self.rays.shape
            return round(N_images * hw * self.sample_ratio / self.batch_size)
        elif self.split == "val":
            return len(self.poses)

    def __getitem__(self, idx):
        if self.split == "train":
            # training pose is retrieved in train_nerf.py
            if self.ray_sampling_strategy == 'all_images': # randomly select images
                img_idxs = np.random.choice(len(self.poses), self.batch_size)
            elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
                img_idxs = np.random.choice(len(self.poses), 1)[0]
            # randomly select pixels
            pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
            rays = self.rays[img_idxs, pix_idxs]
            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                      'color': rays[:, :3]}

        elif self.split == "val":
            rays = self.rays[idx]
            sample = {'pose': self.poses[idx], 'img_idxs': idx,
                      'color': rays[:, :3]}

        return sample
