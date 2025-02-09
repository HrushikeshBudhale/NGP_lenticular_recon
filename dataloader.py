import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import utils
import imageio.v2 as imageio
import json


class BaseDataloader:
    def __init__(self,):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.N, self.images, self.c2w, self.Ks = {}, {}, {}, {}
        self.focal, self.H, self.W = None, None, None

    def show_image(self, index, stype="train") -> None:
        image, pose = self.images[stype][index], self.c2w[stype][index]
        print("image shape: ", image.shape)
        print("pose: ", pose)
        image = (image * 255).astype(np.uint8)
        plt.imshow(image)
        plt.show()

    def show_cameras(self, stype="train") -> None:
        poses = self.c2w[stype]
        utils.view_poses(poses)
    
    def get_data(self, stype="train") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = torch.Tensor(self.images[stype]).to(self.device)
        poses  = torch.Tensor(self.c2w[stype]).to(self.device)
        Ks     = self.Ks[stype].to(self.device)
        return images, poses, Ks
    
    def set_test_data(self, poses, Ks=None) -> None:
        self.c2w["test"] = poses
        self.N["test"] = len(poses)
        if Ks is None:
            K = utils.intrinsic_matrix(self.focal, self.focal, self.W / 2, self.H / 2)
            Ks = K.unsqueeze(0).repeat(self.N["test"], 1, 1)                                 # (N_test, 3, 3)
        self.Ks["test"] = Ks
        self.images["test"] = np.zeros((self.N["test"], self.H, self.W, 3))                 # (N_test, H, W, 3)


class TinyNerfDataloader(BaseDataloader):
    # tested on 200x200 image resolution
    def __init__(self, path):
        super().__init__()
        data = np.load(path)
        
        self.c2w["train"]    = data["c2ws_train"]                                           # (N_train, 4, 4)
        self.c2w["val"]      = data["c2ws_val"]                                             # (N_val, 4, 4)
        self.c2w["test"]     = data["c2ws_test"]                                            # (N_test, 4, 4)
        self.N["train"]      = self.c2w["train"].shape[0]
        self.N["val"]        = self.c2w["val"].shape[0]
        self.N["test"]       = self.c2w["test"].shape[0]
        self.images["train"] = data["images_train"] / 255.0                                 # (N_train, H, W, 3)
        self.images["val"]   = data["images_val"]   / 255.0                                 # (N_val, H, W, 3)
        self.images["test"]  = np.zeros((self.N["test"], *self.images["val"].shape[1:]))    # (N_test, H, W, 3)

        N_images = self.N["train"] + self.N["val"] + self.N["test"]
        self.H, self.W = self.images["train"].shape[1:3]
        self.focal = data["focal"].item()                                                        # float
        K = utils.intrinsic_matrix(self.focal, self.focal, self.W / 2, self.H / 2)
        Ks = K.unsqueeze(0).repeat(N_images, 1, 1)
        self.Ks["train"] = Ks[:self.N["train"]]                                             # (N_train, 3, 3)
        self.Ks["val"]   = Ks[self.N["train"]:self.N["train"]+self.N["val"]]                # (N_val, 3, 3)
        self.Ks["test"]  = Ks[self.N["train"]+self.N["val"]:]                               # (N_test, 3, 3)


class CustomDataloader(BaseDataloader):
    # tested on 100x100 res image
    def __init__(self, path):
        super().__init__()
        data = np.load(path)
        images: torch.Tensor = data["images"]
        poses:  torch.Tensor = data["poses"]
        N_images, self.H, self.W = images.shape[:3]
        
        # randomize
        idx = np.random.permutation(N_images)
        images, poses = images[idx], poses[idx]
        
        # split train, val, test
        self.N["train"]      = int(0.8 * N_images)
        self.N["val"]        = int(0.15 * N_images)
        self.N["test"]       = N_images - self.N["train"] - self.N["val"]
        self.images["train"] = images[:self.N["train"]]                                     # (N_train, H, W, 3)
        self.images["val"]   = images[self.N["train"]:self.N["train"]+self.N["val"]]        # (N_val, H, W, 3)
        self.images["test"]  = images[self.N["train"]+self.N["val"]:]                       # (N_test, H, W, 3)
        self.c2w["train"]    = poses[:self.N["train"]]                                      # (N_train, 4, 4)
        self.c2w["val"]      = poses[self.N["train"]:self.N["train"]+self.N["val"]]         # (N_val, 4, 4)
        self.c2w["test"]     = poses[self.N["train"]+self.N["val"]:]                        # (N_test, 4, 4)
        
        self.c2w["train"][:, :, 1:3] = -self.c2w["train"][:, :, 1:3]
        self.c2w["val"][:, :, 1:3]   = -self.c2w["val"][:, :, 1:3]
        self.c2w["test"][:, :, 1:3]  = -self.c2w["test"][:, :, 1:3]
        
        self.focal = data["focal"].item()                                                        # float
        K = utils.intrinsic_matrix(self.focal, self.focal, self.W / 2, self.H / 2)
        Ks = K.unsqueeze(0).repeat(N_images, 1, 1)
        self.Ks["train"]     = Ks[:self.N["train"]]                                         # (N_train, 3, 3)
        self.Ks["val"]       = Ks[self.N["train"]:self.N["train"]+self.N["val"]]            # (N_val, 3, 3)
        self.Ks["test"]      = Ks[self.N["train"]+self.N["val"]:]                           # (N_test, 3, 3)


class BlenderDataloader(BaseDataloader):
    # tested on 800x800 res image 
    def __init__(self, base_dir: str, scale:float=1.0, testskip: int=1):
        super().__init__()

        for s in ['train', 'val', 'test']:
            skip = 1 if s=='train' or testskip==0 else testskip
            with open(os.path.join(base_dir, f'transforms_{s}.json'), 'r') as fp:
                meta = json.load(fp)
            imgs, poses = [], []
            for frame in meta['frames'][::skip]:
                fname = os.path.join(base_dir, frame['file_path'] + '.png')
                img = imageio.imread(fname)
                img = img[:,:,:3]               # remove alpha channel
                imgs.append(cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
                poses.append(np.array(frame['transform_matrix']))
            self.images[s] = np.array(imgs).astype(np.float32) / 255.0
            self.c2w[s] = np.array(poses).astype(np.float32)
            # Rotate 180 degrees along X axis to bring in colmap convention
            self.c2w[s][:, :, 1:3] = -self.c2w[s][:, :, 1:3]
            self.N[s] = len(imgs)
            self.H, self.W = self.images[s][0].shape[:2]
            camera_angle_x = float(meta['camera_angle_x'])
            self.focal = .5 * self.W / np.tan(.5 * camera_angle_x)
            K = utils.intrinsic_matrix(self.focal, self.focal, self.W / 2, self.H / 2)
            self.Ks[s] = K.unsqueeze(0).repeat(self.N[s], 1, 1)

class ImageReconDataloader(BaseDataloader):
    def __init__(self, image_path: str, scale:float=1):
        super().__init__()

        image = imageio.imread(image_path)
        H, W = image.shape[:2]
        focal = W / 2
        pose = np.eye(4)
        pose[2,3] = -1 # set camera at focal length distance
        for s in ['train', 'val', 'test']:
            self.images[s] = np.array([image[...,:3]]).astype(np.float32) / 255.0
            self.c2w[s] = np.array([pose]).astype(np.float32)
            self.N[s] = 1
            self.H, self.W = H, W
            self.focal = focal
            K = utils.intrinsic_matrix(self.focal, self.focal, self.W / 2, self.H / 2)
            self.Ks[s] = K.unsqueeze(0).repeat(self.N[s], 1, 1)

class RaysData:
    def __init__(self, images: torch.Tensor, poses: torch.Tensor, Ks: torch.Tensor):
        self.K = Ks[0]
        self.N_images, self.H, self.W = images.shape[:3]

        # create uv grid
        vs, us = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), indexing='ij')
        uv = torch.stack([us, vs], dim=-1).to(images.device).float()
        uv = uv + 0.5 # add 0.5 offset to each pixel
        self.uv_flattened = uv.reshape(-1, 2)                               # (H*W, 2)

        r_o, r_d = utils.pixel_to_ray(self.uv_flattened, poses, Ks)         # (N, H*W, 3), (N, H*W, 3)
        self.pixels = images.reshape(-1, 3)                                 # (N*H*W, 3)
        self.r_o_flattened = r_o.reshape(-1, 3)                             # (N*H*W, 3)
        self.r_d_flattened = r_d.reshape(-1, 3)                             # (N*H*W, 3)

    def cast_rays(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.pixels.shape[0], (batch_size,), device=self.pixels.device)
        return self.r_o_flattened[idx], self.r_d_flattened[idx], self.pixels[idx] # (B, 3), (B, 3), (B, 3)

    def cast_image_rays(self, image_index: int=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = np.random.randint(self.N_images) if image_index is None else image_index
        start_idx = index * self.H * self.W
        end_idx = start_idx + self.H * self.W
        r_o = self.r_o_flattened[start_idx:end_idx]
        r_d = self.r_d_flattened[start_idx:end_idx]
        pixels = self.pixels[start_idx:end_idx]
        return r_o, r_d, pixels # (B, 3), (B, 3), (B, 3)
    

# dataloader factory
def get_dataloader(type: str, data_path: str) -> BaseDataloader:
    loaders = {
        "tiny_nerf": TinyNerfDataloader,
        "custom": CustomDataloader,
        "blender": BlenderDataloader,
        "ImageRecon": ImageReconDataloader,
    }
    if type not in loaders:
        raise ValueError(f"Invalid dataloader type: {type}")
    return loaders[type](data_path)


if __name__ == "__main__":
    split_type = "train"
    # dataloader = get_dataloader("custom", "data/lego_100x100.npz")
    # dataloader = get_dataloader("tiny_nerf", "data/lego_200x200.npz")
    # dataloader = get_dataloader("blender", "data/lego")
    dataloader = get_dataloader("ImageRecon", "data/lego/train/r_2.png")
    print(f"number of {split_type} images: {dataloader.N[split_type]}")
    for i in range(dataloader.N[split_type])[:3]:
        dataloader.show_image(index=i, stype=split_type)
    dataloader.show_cameras(split_type)