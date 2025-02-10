import os
import yaml
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_yaml(path):
    with open(path, 'r') as stream:
        return yaml.safe_load(stream)


def split_batch(tensors: list[torch.Tensor], batch_size: int) -> list[torch.Tensor]:
    # tensors: (N_rays, ...)
    
    # assert if all tensors have same shape[0]
    N_rays = tensors[0].shape[0]
    assert all(tensor.shape[0] == N_rays for tensor in tensors)    
    
    batches = [[ t[i:i+batch_size] for t in tensors] for i in range(0, N_rays, batch_size)]
    return batches  # (N_batches, N_tensors, N_rays, ...)


def create_gif(image_folder:str, gif_name:str, duration:float=5.0):
    frames = [imageio.imread(image_folder + image_name) for image_name in os.listdir(image_folder)]
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration, loop=2)


def view_poses(poses: torch.Tensor) -> None:
    R, T = poses[:, :3, :3], poses[:, :3, 3]
    Z = R[:, :, 2]
    figure = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')
    _ = figure.quiver(T[:, 0], T[:, 1], T[:, 2], Z[:, 0], Z[:, 1], Z[:, 2], length=0.2)
    figure.set_xlabel('x')
    figure.set_ylabel('y')
    figure.set_zlabel('z')
    plt.axis('equal')
    plt.show()


def save_psnr_plot(psnr_score:list) -> None:
    # psnr_score: (N_vals, 2)
    psnr_score = np.array(psnr_score)
    plt.figure()
    plt.plot(psnr_score[:, 0], psnr_score[:, 1])
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR Score")
    plt.savefig("psnr_score.png")
    plt.close()


def save_checkpoint(epoch, scores, models, optimizer, ckpt_path:str) -> None:
    torch.save({
        'epoch':epoch,
        'scores': scores,
        'optimizer_state_dict': optimizer.state_dict(),
        'model_state_dicts': {k: v.state_dict() for k, v in models.items()}
    }, ckpt_path)


def load_checkpoint(models, optimizer, ckpt_path:str):
    if not os.path.exists(ckpt_path):
        return 1, []
    
    checkpoint=torch.load(ckpt_path)
    resume_epoch = checkpoint['epoch']
    psnr_scores = checkpoint['scores']
    for k, v in models.items():
        v.load_state_dict(checkpoint['model_state_dicts'][k])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f'Loaded checkpoint: {resume_epoch}')
    return resume_epoch, psnr_scores


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n.split(".")[0])
            ave_grads.append(p.grad.abs().mean().cpu().numpy())
            max_grads.append(p.grad.abs().max().cpu().numpy())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

def intrinsic_matrix(fx, fy, cx, cy) -> torch.Tensor:
    return torch.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    img1, img2 = img1.detach().clone(), img2.detach().clone()
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return (20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))).item()


def pixel_to_ray(uv, poses, Ks, is_training=True) -> tuple[torch.Tensor, torch.Tensor]:
    # uv: (H*W, 2)
    # poses: (N, 4, 4)
    # Ks: (N, 3, 3)
    uv = uv.unsqueeze(0).repeat(len(Ks), 1, 1)                              # (N, H*W, 2)
    xc = pixel_to_camera(uv, Ks)                                            # (N, H*W, 3)
    xw = camera_to_world(xc, poses)                                         # (N, H*W, 3)
    if is_training:
        eye = torch.eye(4, device=uv.device)
        eye[[0,1],:2] *= -1 # Rotate camera 180 degrees along Z
        eye = eye.unsqueeze(0).repeat(len(Ks), 1, 1)                        # (N, 4,4)
        xw = camera_to_world(xc, eye)                                       # (N, H*W, 3)

    # ray origins
    r_o = poses[:, :3, -1].unsqueeze(1)                                     # (N, 1, 3)
    r_o = r_o.repeat(1, uv.shape[1], 1)                                     # (N, H*W, 3)

    # ray directions
    r_d = xw - r_o                                                          # (N, H*W, 3)
    return r_o, r_d                                                         # (N, H*W, 3), (N, H*W, 3)


def pixel_to_camera(uv, Ks) -> torch.Tensor:
    # Computes camera coordinates from pixel coordinates
    # uv: (H*W, 2)
    # Ks: (N, 3, 3)
    uv_homogeneous = torch.cat([uv, torch.ones_like(uv[...,:1])], dim=-1)   # (N, H*W, 3)
    uv_homogeneous = uv_homogeneous.permute(0, 2, 1)                        # (N, 3, H*W)
    K_inv = torch.inverse(Ks)                                               # (N, 3, 3)
    xc_homogeneous = K_inv.bmm(uv_homogeneous)                              # (N, 3, H*W)
    xc = xc_homogeneous.permute(0, 2, 1)                                    # (N, H*W, 3)
    return xc                                                               # (N, H*W, 3)


def camera_to_world(xc, poses) -> torch.Tensor:
    # xc: (N, H*W, 3)
    # poses: (N, 4, 4)
    xc_homogeneous = torch.cat([xc, torch.ones_like(xc[...,:1])], dim=-1)   # (N, H*W, 4)
    xc_homogeneous = xc_homogeneous.permute(0, 2, 1)                        # (N, 4, H*W)
    xw_homogeneous = poses.bmm(xc_homogeneous)                              # (N, 4, H*W)
    xw = xw_homogeneous.permute(0, 2, 1)[...,:3]                            # (N, H*W, 3)
    return xw                                                               # (N, H*W, 3)