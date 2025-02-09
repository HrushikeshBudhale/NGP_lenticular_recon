import torch


class Renderer:
    def __init__(self, conf: dict):
        self.near = conf["near"]
        self.far = conf["far"]
        self.T_thresh = conf["t_thresh"] # Transmittance threshold
    
    def calc_weights(self, sigmas: torch.Tensor, z_vals: torch.Tensor) -> torch.Tensor:
        # sigmas: (N_rays, N_samples, 1)
        # z_vals: (N_rays, N_samples)
        N_rays = sigmas.shape[0]
        
        dists = (z_vals[:,1:] - z_vals[:,:-1]).unsqueeze(-1)                                              # (N_rays, N_samples-1, 1)
        dists = torch.cat([dists, 1e10 * torch.ones_like(sigmas[:, :1, :])], dim=1)                       # (N_rays, N_samples, 1)
        alpha = 1 - torch.exp(-sigmas * dists)                                                            # (N_rays, N_samples, 1)
        
        # Transmittance: Cumulative product expresses the probability of ray transmitting through each sample
        T_i = torch.cumprod((1-alpha),1)                                                                  # (N_rays, N_samples, 1)
        T_i = torch.where(T_i < self.T_thresh, torch.zeros_like(T_i), T_i) # Early stop: Set other sample contributions to 0
        T_i = torch.cat([torch.ones((N_rays, 1, 1), device=sigmas.device), T_i[:, :-1]], dim=1)           # (N_rays, N_samples, 1)
        weights = alpha * T_i
        return weights # (N_rays, N_samples, 1)

    def volume_render(self, rgb, sigmas, z_vals, pts_mask) -> torch.Tensor:
        # rgb:    (N_rays, N_samples, 3), sigmas: (N_rays, N_samples, 1)
        # z_vals: (N_rays, N_samples), pts_mask: (N_rays, N_samples)
        N_rays = sigmas.shape[0]
        rendered_colors = torch.zeros((N_rays, 3), device=sigmas.device)
        rendered_depths = torch.zeros((N_rays, 3), device=sigmas.device)
        valid_rays = torch.any(pts_mask, dim=1).flatten()                                                 # (Np_rays,)
        weights = self.calc_weights(sigmas[valid_rays], z_vals[valid_rays])                               # (Np_rays, N_samples, 1)
        z_vals = (z_vals[valid_rays] - self.near) / (self.far - self.near)
        z_vals = z_vals.unsqueeze(-1).repeat(1, 1, 3)                                                     # (Np_rays, N_samples, 3)
        rendered_depths[valid_rays] = torch.sum(weights * z_vals, dim=1)                                  # (N_rays, N_samples, 3)
        rendered_colors[valid_rays] = torch.sum(weights * rgb[valid_rays], dim=1) 
        return rendered_colors, rendered_depths                                                           # (N_rays, N_samples, 3), (N_rays, N_samples, 3)