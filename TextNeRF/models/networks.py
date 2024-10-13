import torch
from torch import nn
import tinycudann as tcnn
import vren
from einops import rearrange
from .custom_functions import TruncExp
import numpy as np

from .rendering import NEAR_DISTANCE


class XYZEncoder(nn.Module):
    def __init__(self, scale, L=16, F=2, log2_T=19, N_min=16):
        super().__init__()
        b = np.exp(np.log(2048 * scale / N_min) / (L - 1))
        self.xyz_encoder = \
            tcnn.NetworkWithInputEncoding(
                n_input_dims=3, n_output_dims=16,
                encoding_config={
                    "otype": "Grid",
                    "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                }
            )

    def forward(self, x):
        h = self.xyz_encoder(x)
        sigmas = TruncExp.apply(h[:, 0])
        return sigmas, h


class ColorRenderer_G(nn.Module):
    def __init__(self, appearance_embed_dim=32):
        self.appearance_embed_dim = appearance_embed_dim
        super().__init__()
        self.rgb_net = tcnn.NetworkWithInputEncoding(
            n_input_dims=3 + 16 + self.appearance_embed_dim,
            n_output_dims=3,
            encoding_config={
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": 3,
                        "otype": "SphericalHarmonics",
                        "degree": 4,
                    },
                    {
                        "n_dims_to_encode": 16 + self.appearance_embed_dim,
                        "otype": "Identity"
                    }
                ]
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": 'Sigmoid',
                "n_neurons": 64,
                "n_hidden_layers": 3
            }
        )

    def forward(self, d, h, appearance_embedding):
        d = d / torch.norm(d, dim=1, keepdim=True)
        d = (d + 1) / 2
        if len(appearance_embedding.shape) == 1:
            appearance_embedding = appearance_embedding.expand(h.shape[0], -1)
        colors = self.rgb_net(torch.cat([d, h, appearance_embedding], 1))
        return colors


class TextNGP_G(nn.Module):
    def __init__(self, scale, num_images, xyz_encoder, color_renderer):
        super().__init__()
        # scene bounding box
        self.scale = scale
        self.appearance_embed_dim = 32
        self.appearance_embed = nn.Embedding(num_images, self.appearance_embed_dim)  # embedding_dim = 32, hard code

        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*scale)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1+int(np.ceil(np.log2(2*scale))), 1)
        self.grid_size = 128
        self.register_buffer('density_bitfield',
            torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8))

        # constants
        L = 16; F = 2; log2_T = 19; N_min = 16
        b = np.exp(np.log(2048*scale/N_min)/(L-1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.xyz_encoder = xyz_encoder
        self.color_renderer = color_renderer

    @property
    def device(self):
        return self.density_bitfield.device

    def density(self, x, return_feat=False):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min)
        sigmas, h = self.xyz_encoder(x)
        if return_feat: return sigmas, h
        return sigmas

    def get_appearance_embedding(self, img_idxs=None):
        if img_idxs is not None:
            appearance_embedding = self.appearance_embed(img_idxs)
        else:
            appearance_embedding = torch.mean(self.appearance_embed.weight, dim=0)
        return appearance_embedding

    def forward(self, x, d, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            colors: (N, 3)
        """
        sigmas, h = self.density(x, return_feat=True)
        colors = self.color_renderer(d, h, kwargs['appearance_embedding'])
        return sigmas, colors

    @torch.no_grad()
    def get_all_cells(self):
        """
        Get all cells from the density grid.
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        indices = vren.morton3D(self.grid_coords).long()
        cells = [(indices, self.grid_coords)] * self.cascades

        return cells

    @torch.no_grad()
    def sample_uniform_and_occupied_cells(self, M, density_threshold):
        """
        Sample both M uniform and occupied cells (per cascade)
        occupied cells are sample from cells with density > @density_threshold
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        cells = []
        for c in range(self.cascades):
            # uniform cells
            coords1 = torch.randint(self.grid_size, (M, 3), dtype=torch.int32,
                                    device=self.density_grid.device)
            indices1 = vren.morton3D(coords1).long()
            # occupied cells
            indices2 = torch.nonzero(self.density_grid[c]>density_threshold)[:, 0]
            # if len(indices2)>0:
            #     rand_idx = torch.randint(len(indices2), (M,),
            #                              device=self.density_grid.device)
            #     indices2 = indices2[rand_idx]
            # coords2 = vren.morton3D_invert(indices2.int())
            # # concatenate
            # cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]
            
            if len(indices2)>0:
                rand_idx = torch.randint(len(indices2), (M,),
                                         device=self.density_grid.device)
                indices2 = indices2[rand_idx]
                coords2 = vren.morton3D_invert(indices2.int())
                # concatenate
                cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]
            else:
                cells += [(indices1, coords1)]

        return cells

    @torch.no_grad()
    def mark_invisible_cells(self, K, poses, img_wh, chunk=64**3):
        """
        mark the cells that aren't covered by the cameras with density -1
        only executed once before training starts

        Inputs:
            K: (3, 3) camera intrinsics
            poses: (N, 3, 4) camera to world poses
            img_wh: image width and height
            chunk: the chunk size to split the cells (to avoid OOM)
        """
        N_cams = poses.shape[0]
        self.count_grid = torch.zeros_like(self.density_grid)
        w2c_R = rearrange(poses[:, :3, :3], 'n a b -> n b a') # (N_cams, 3, 3)
        w2c_T = -w2c_R@poses[:, :3, 3:] # (N_cams, 3, 1)
        cells = self.get_all_cells()
        for c in range(self.cascades):
            indices, coords = cells[c]
            for i in range(0, len(indices), chunk):
                xyzs = coords[i:i+chunk]/(self.grid_size-1)*2-1
                s = min(2**(c-1), self.scale)
                half_grid_size = s/self.grid_size
                xyzs_w = (xyzs*(s-half_grid_size)).T # (3, chunk)
                xyzs_c = w2c_R @ xyzs_w + w2c_T # (N_cams, 3, chunk)
                uvd = K @ xyzs_c # (N_cams, 3, chunk)
                uv = uvd[:, :2]/uvd[:, 2:] # (N_cams, 2, chunk)
                in_image = (uvd[:, 2]>=0)& \
                           (uv[:, 0]>=0)&(uv[:, 0]<img_wh[0])& \
                           (uv[:, 1]>=0)&(uv[:, 1]<img_wh[1])
                covered_by_cam = (uvd[:, 2]>=NEAR_DISTANCE)&in_image # (N_cams, chunk)
                # if the cell is visible by at least one camera
                self.count_grid[c, indices[i:i+chunk]] = \
                    count = covered_by_cam.sum(0)/N_cams

                too_near_to_cam = (uvd[:, 2]<NEAR_DISTANCE)&in_image # (N, chunk)
                # if the cell is too close (in front) to any camera
                too_near_to_any_cam = too_near_to_cam.any(0)
                # a valid cell should be visible by at least one camera and not too close to any camera
                valid_mask = (count>0)&(~too_near_to_any_cam)
                self.density_grid[c, indices[i:i+chunk]] = \
                    torch.where(valid_mask, 0., -1.)

    @torch.no_grad()
    def update_density_grid(self, density_threshold, warmup=False, decay=0.95, erode=False):
        density_grid_tmp = torch.zeros_like(self.density_grid)
        if warmup: # during the first steps
            cells = self.get_all_cells()
        else:
            cells = self.sample_uniform_and_occupied_cells(self.grid_size**3//4,
                                                           density_threshold)
        # infer sigmas
        for c in range(self.cascades):
            indices, coords = cells[c]
            s = min(2**(c-1), self.scale)
            half_grid_size = s/self.grid_size
            xyzs_w = (coords/(self.grid_size-1)*2-1)*(s-half_grid_size)
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w)*2-1) * half_grid_size
            density_grid_tmp[c, indices] = self.density(xyzs_w)

        if erode:
            # My own logic. decay more the cells that are visible to few cameras
            decay = torch.clamp(decay**(1/self.count_grid), 0.1, 0.95)
        self.density_grid = \
            torch.where(self.density_grid<0,
                        self.density_grid,
                        torch.maximum(self.density_grid*decay, density_grid_tmp))

        mean_density = self.density_grid[self.density_grid>0].mean().item()

        vren.packbits(self.density_grid, min(mean_density, density_threshold),
                      self.density_bitfield)



class TextNeRF(nn.Module):
    def __init__(self, ngp, e_attr_output_dims):
        super().__init__()
        self.ngp = ngp
        for param in self.ngp.xyz_encoder.parameters():  # freeze geometry related component
            param.requires_grad = False
        self.ngp.xyz_encoder.eval()

        self.color_decorator = \
            tcnn.Network(
                n_input_dims=ngp.xyz_encoder.n_output_dims + e_attr_output_dims,
                n_output_dims=ngp.xyz_encoder.n_output_dims,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": 2,
                }
            )
        self.scale = self.ngp.scale
        self.register_buffer('center', self.ngp.center)
        self.register_buffer('xyz_min', self.ngp.xyz_min)
        self.register_buffer('xyz_max', self.ngp.xyz_max)
        self.register_buffer('half_size', self.ngp.half_size)

        self.cascades = self.ngp.cascades
        self.grid_size = self.ngp.grid_size
        self.register_buffer('density_bitfield', self.ngp.density_bitfield)
        
    @property
    def device(self):
        return self.ngp.device

    def forward(self, x, d, **kwargs):
        return self.ngp(x, d, **kwargs)
