import os
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataloader import RaysData, get_dataloader
from ray_sampling import RaySampler
from renderer import Renderer
from model import NGP
import utils
from gen_extrinsics import gen_circular_poses, gen_surrounding_poses

if __name__ == "__main__":
    os.makedirs("rgb_output", exist_ok=True)
    os.makedirs("depth_output", exist_ok=True)
    utils.set_seed(60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conf = utils.load_yaml("conf.yaml")

    dataloader = get_dataloader(conf["dataset_type"], conf["dataset_path"])
    test_rays = RaysData(*dataloader.get_data(stype="test"))
    ray_sampler = RaySampler(conf)
    renderer = Renderer(conf)

    model = {'nerf':NGP(config=conf, device=device)}
    utils.load_checkpoint(model, None, conf["ckpt_path"])
    model['nerf'].eval()
    with torch.no_grad():
        for i in tqdm(range(test_rays.N_images)):
            rays_o, rays_d, rays_rgb = test_rays.cast_image_rays(image_index=i)
            
            # perform batched inference
            comp_rgbs, comp_depths = [], []
            for (b_rays_o, b_rays_d) in utils.split_batch((rays_o, rays_d), conf["rays_per_batch"]):
                points, z_vals = ray_sampler.sample_along_rays(b_rays_o, b_rays_d)
                rays_dn = b_rays_d.unsqueeze(-2).repeat(1, ray_sampler.N_samples, 1)
                rgb, sigmas, pts_mask = model["nerf"](points, rays_dn)
                comp_rgb, comp_depth = renderer.volume_render(rgb, sigmas, z_vals, pts_mask)
                comp_depths.append(comp_depth)
                comp_rgbs.append(comp_rgb)
            comp_rgb = torch.cat(comp_rgbs, dim=0)
            comp_depth = torch.cat(comp_depths, dim=0)

            image = comp_rgb.reshape(test_rays.H, test_rays.W, 3).cpu().numpy()
            depth = comp_depth.reshape(test_rays.H, test_rays.W, 3).cpu().numpy()
            plt.imsave(f"rgb_output/{i:04}.png", image)
            plt.imsave(f"depth_output/{i:04}.png", depth)
    utils.create_gif("rgb_output/", "rgb_output.gif")
    utils.create_gif("depth_output/", "depth_output.gif")
    shutil.rmtree('rgb_output')
    shutil.rmtree('depth_output')