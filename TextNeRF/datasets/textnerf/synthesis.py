import os
from ..utils.ray_utils import *
from ..utils.colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary


class TextNeRFSynthesisDataset(object):
    """ Geometry dataset 以 ray 为 sample 粒度 """
    def __init__(self, root_dir, downsample=1.0, **kwargs):
        self.root_dir = root_dir
        self.downsample = downsample
        self.color_mode = kwargs.get("color_mode", "RGB")

        self.read_intrinsics()
        self.read_meta(**kwargs)

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
        self.K = np.asarray([[fx, 0, cx],
                             [0, fy, cy],
                             [0,  0,  1]])
        self.directions = get_ray_directions(h, w, self.K)

    def read_meta(self, **kwargs):
        """ 整理好所有需要用于训练的样本集合，这里并没有一下子整理出所有的光线，而是整理出了所有ray的gt，和pose """
        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
        img_names = [imdata[k].name for k in imdata]
        perm = np.argsort(img_names)

        # read successfully reconstructed images and ignore others
        self.img_paths = [os.path.join(self.root_dir, 'images', name) for name in sorted(img_names)]
        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])
        # (x, y) -> pts3D_id
        pts3D_id_to_xys = []
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat(); t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]

            pts3D_id_to_xys += [{p3d_id: tuple(xy.astype(int).tolist()) for xy, p3d_id in zip(im.xys, im.point3D_ids)}]

        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[perm, :3] # (N_images, 3, 4) cam2world matrices

        pts3D_id_to_xys = [pts3D_id_to_xys[i] for i in perm]

        pts3d_data = read_points3d_binary(os.path.join(self.root_dir, 'sparse/0/points3D.bin'))
        pts3d = np.array([pts3d_data[k].xyz for k in pts3d_data]) # (N, 3)
        self.poses, self.pts3d = center_poses(poses, pts3d)
        scale = np.linalg.norm(self.poses[..., 3], axis=-1).min()
        self.poses[..., 3] /= scale
        self.pts3d /= scale

        self.training_embed_size = len([x for i, x in enumerate(self.img_paths) if i%8!=0])
        self.img_paths = [x for _, x in enumerate(self.img_paths)]
        self.poses = np.array([x for _, x in enumerate(self.poses)])

        self.pts3D_id_to_xys = pts3D_id_to_xys  # list of dict using tuple as key

    def __len__(self):
        return len(self.poses)


