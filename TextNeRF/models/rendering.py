import torch
from .custom_functions import \
    RayAABBIntersector, RayMarcher, VolumeRenderer
from einops import rearrange
import vren

MAX_SAMPLES = 1024
NEAR_DISTANCE = 0.01


@torch.cuda.amp.autocast()
def render(model, rays_o, rays_d, **kwargs):
    """
    Render rays by
    1. Compute the intersection of the rays with the scene bounding box
    2. Follow the process in @render_func (different for train/test)

    Inputs:
        model: NGP
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions

    Outputs:
        result: dictionary containing final color and depth
    """
    rays_o = rays_o.contiguous()
    rays_d = rays_d.contiguous()
    _, hits_t, _ = \
        RayAABBIntersector.apply(rays_o, rays_d, model.center, model.half_size, 1)
    hits_t[(hits_t[:, 0, 0]>=0)&(hits_t[:, 0, 0]<NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE

    if kwargs.get('test_time', False):
        render_func = __render_rays_test
    else:
        render_func = __render_rays_train

    results = render_func(model, rays_o, rays_d, hits_t, **kwargs)
    for k, v in results.items():
        if kwargs.get('to_cpu', False):
            v = v.cpu()
            if kwargs.get('to_numpy', False):
                v = v.numpy()
        results[k] = v
    return results


@torch.no_grad()
def __render_rays_test(model, rays_o, rays_d, hits_t, **kwargs):
    """
    Render rays by

    while (a ray hasn't converged)
        1. Move each ray to its next occupied @N_samples (initially 1) samples 
           and evaluate the properties (sigmas, colors) there
        2. Composite the result to output; if a ray has transmittance lower
           than a threshold, mark this ray as converged and stop marching it.
           When more rays are dead, we can increase the number of samples
           of each marching (the variable @N_samples)
    """
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    results = {}

    # output tensors to be filled in
    N_rays = len(rays_o)
    device = rays_o.device
    opacity = torch.zeros(N_rays, device=device)
    depth = torch.zeros(N_rays, device=device)
    color = torch.zeros(N_rays, 3, device=device)

    samples = total_samples = 0
    alive_indices = torch.arange(N_rays, device=device)
    # if it's synthetic data, bg is majority so min_samples=1 effectively covers the bg
    # otherwise, 4 is more efficient empirically
    min_samples = 1 if exp_step_factor==0 else 4

    while samples < kwargs.get('max_samples', MAX_SAMPLES):
        N_alive = len(alive_indices)
        if N_alive==0: break

        # the number of samples to add on each ray
        N_samples = max(min(N_rays//N_alive, 64), min_samples)
        samples += N_samples

        xyzs, dirs, deltas, ts, N_eff_samples = \
            vren.raymarching_test(rays_o, rays_d, hits_t[:, 0], alive_indices,
                                  model.density_bitfield, model.cascades,
                                  model.scale, exp_step_factor,
                                  model.grid_size, MAX_SAMPLES, N_samples)

        total_samples += N_eff_samples.sum()
        n1, n2, c = dirs.shape
        xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
        if kwargs.get("new_d", None) is not None:
            dirs = kwargs.pop("new_d")[:, None, :].expand(n1, n2, c)
        dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
        
        valid_mask = ~torch.all(dirs==0, dim=1)
        if valid_mask.sum()==0: break
        
        sigmas = torch.zeros(len(xyzs), device=device)
        colors = torch.zeros(len(xyzs), 3, device=device)
        sigmas[valid_mask], _colors = model(xyzs[valid_mask], dirs[valid_mask], **kwargs)
        colors[valid_mask] = _colors.float()
        sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
        colors = rearrange(colors, '(n1 n2) c -> n1 n2 c', n2=N_samples)

        vren.composite_test_fw(
            sigmas, colors, deltas, ts,
            hits_t[:, 0], alive_indices, kwargs.get('T_threshold', 1e-4),
            N_eff_samples, opacity, depth, color)
        alive_indices = alive_indices[alive_indices>=0] # remove converged rays

    results['opacity'] = opacity
    results['depth'] = depth
    results['color'] = color
    results['total_samples'] = total_samples # total samples for all rays

    if exp_step_factor==0: # synthetic
        color_bg = torch.ones(3, device=device)
    else: # real
        color_bg = torch.zeros(3, device=device)
    results['color'] += color_bg*rearrange(1-opacity, 'n -> n 1')

    return results


def __render_rays_train(model, rays_o, rays_d, hits_t, **kwargs):
    """
    Render rays by
    1. March the rays along their directions, querying @density_bitfield
       to skip empty space, and get the effective sample points (where
       there is object)
    2. Infer the NN at these positions and view directions to get properties
       (currently sigmas and colors)
    3. Use volume rendering to combine the result (front to back compositing
       and early stop the ray if its transmittance is below a threshold)
    """
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    results = {}
    if kwargs.get('color_decorator', None) is not None:
        with torch.no_grad():
            (rays_a, xyzs, dirs,
             results['deltas'], results['ts'], results['rm_samples']) = \
                RayMarcher.apply(
                    rays_o, rays_d, hits_t[:, 0], model.density_bitfield,
                    model.cascades, model.scale,
                    exp_step_factor, model.grid_size, MAX_SAMPLES)
    else:
        (rays_a, xyzs, dirs,
        results['deltas'], results['ts'], results['rm_samples']) = \
            RayMarcher.apply(
                rays_o, rays_d, hits_t[:, 0], model.density_bitfield,
                model.cascades, model.scale,
                exp_step_factor, model.grid_size, MAX_SAMPLES)

        for k, v in kwargs.items(): # supply additional inputs, repeated per ray
            if isinstance(v, torch.Tensor):
                kwargs[k] = torch.repeat_interleave(v[rays_a[:, 0]], rays_a[:, 2], 0)

    sigmas, colors = model(xyzs, dirs, **kwargs)

    (results['vr_samples'], results['opacity'],
    results['depth'], results['color'], results['ws']) = \
        VolumeRenderer.apply(sigmas, colors.contiguous(), results['deltas'], results['ts'],
                             rays_a, kwargs.get('T_threshold', 1e-4))
    results['rays_a'] = rays_a

    if exp_step_factor==0: # synthetic
        color_bg = torch.ones(3, device=rays_o.device)
    else: # real
        if kwargs.get('random_bg', False):
            color_bg = torch.rand(3, device=rays_o.device)
        else:
            color_bg = torch.zeros(3, device=rays_o.device)
    results['color'] = results['color'] + \
                     color_bg*rearrange(1-results['opacity'], 'n -> n 1')

    return results
