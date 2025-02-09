import torch

class RaySampler:
    def __init__(self, conf: dict):
        self.near = conf["near"]
        self.far = conf["far"]
        self.N_samples = conf["samples_per_ray"]
        self.perturb_samples = conf["perturb_samples"]
        self.z_vals = torch.linspace(self.near, self.far, self.N_samples)                                       # (N_samples)

    def sample_along_rays(self, r_o: torch.Tensor, r_d: torch.Tensor):
        """Stratified sampling along rays"""
        # r_o: (N_rays, 3)
        # r_d: (N_rays, 3)
        z_vals = self.z_vals.to(r_o.device).unsqueeze(0).repeat(r_o.shape[0], 1)                                # (N_rays, N_samples)
        mids = (z_vals[:, 1:] + z_vals[:, :-1]) / 2.0                                                           # (N_rays, N_samples-1)
        lower = torch.cat([z_vals[:, :1], mids], dim=-1)                                                        # (N_rays, N_samples)
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)                                                       # (N_rays, N_samples)
        z_vals = lower + (upper - lower) * torch.rand(z_vals.shape, device=r_o.device) * self.perturb_samples   # (N_rays, N_samples)
        r_o, r_d = r_o.unsqueeze(1).repeat(1, self.N_samples, 1), r_d.unsqueeze(1).repeat(1, self.N_samples, 1) # (N_rays, N_samples, 3), (N_rays, N_samples, 3)
        points = r_o + r_d * z_vals.unsqueeze(-1)                                                               # (N_rays, N_samples, 3)
        return points, z_vals