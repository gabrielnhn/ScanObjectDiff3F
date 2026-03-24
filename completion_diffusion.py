import gc
import torch
import numpy as np
import math
from tqdm import tqdm
from time import time

from pc_utils import load_scanobjectnn_to_pytorch3d
import dino2
import ip_controlnet

from pytorch3d.renderer import (
    look_at_view_transform,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras
)
from dino2 import get_dino_features_and_score
# from dino3 import get_dino_features_and_score
import torchvision
TPL = torchvision.transforms.ToPILImage
tpl = TPL()

from torchvision.models import ResNet50_Weights
imagenet_classes = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]

FEATURE_DIMS = 768 
# FEATURE_DIMS = 384 


from datetime import datetime
import os
if not os.path.isdir("renders"):
    os.mkdir("renders")


import torch
import torch.nn as nn
from pytorch3d.renderer import AlphaCompositor

class CircleRenderer(nn.Module):
    """
    Render circles, not spheres, requires precomputed fragments
    """
    def __init__(self, background_color=(1.0, 1.0, 1.0)):
        super().__init__()
        self.compositor = AlphaCompositor(background_color=background_color)

    def forward(self, fragments, pcd_batch):
        weights = (fragments.idx != -1).float().permute(0, 3, 1, 2)
        indices = fragments.idx.long().permute(0, 3, 1, 2)
        features = pcd_batch.features_packed().permute(1, 0)
        images = self.compositor(indices, weights, features)
        return images.permute(0, 2, 3, 1)

def get_grid(H, W, device):
    x_range = torch.linspace(1, -1, W, device=device) # X Flip
    y_range = torch.linspace(1, -1, H, device=device) # Y Invert
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
    return torch.stack((x_grid, y_grid), dim=-1)

def render_with_pytorch3d(device, pcd, num_views, points, H=512, W=512):
    bbox = pcd.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bb_diff = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) / 2.0
    
    # distance
    scaling_factor = 0.65
    distance = torch.sqrt((bb_diff * bb_diff).sum())
    distance *= scaling_factor
    
    # trajectory - NOT THE SAME AS IN DIFF3F ANYMORE
    steps = int(math.sqrt(num_views))
    if steps < 1: steps = 1
    azimuth_end = 360 - 360/steps
    azimuth = torch.linspace(start=0, end=azimuth_end, steps=steps)
    azimuth = torch.repeat_interleave(azimuth, steps)
    
    # ELEVATION - now different
    # elevation = torch.linspace(start=0, end=azimuth_end, steps=steps).repeat(steps)
    elev_start, elev_end = 0, 80
    elevation = torch.linspace(start=elev_start, end=elev_end, steps=steps).repeat(steps)
    
    # Calculate R, T
    R, T = look_at_view_transform(
        dist=distance, 
        elev=elevation, 
        azim=azimuth, 
        device=device, 
        at=bbox_center.unsqueeze(0)
    )
    
    cameras = PerspectiveCameras(device=device, R=R, T=T)
    
    # Recalculate actual views (since sqrt might truncate)
    num_actual_views = R.shape[0]


    raster_settings = PointsRasterizationSettings(
        image_size=(H, W), 
        radius=0.02,        
        points_per_pixel=1, 
        bin_size=0 
    )
    
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    
    renderer = CircleRenderer(background_color=(1,1,1)).to(device)

    pcd_batch = pcd.extend(num_actual_views)

    # do the rendering
    fragments = rasterizer(pcd_batch)
    images = renderer(fragments, pcd_batch).cpu()

    depth = fragments.zbuf[..., 0].cpu()
    valid_mask = (fragments.idx[..., 0] != -1).cpu()
    depth[~valid_mask] = -1
    del fragments, pcd_batch #, pt_features
    return images, depth, cameras

def get_diffused_depth(
    device, dino_model, pcd, 
    USE_SCORE,
    ip_pipe,
    num_views=50, H=512, W=512, 
    points=None,
    
):
    t1 = time()
    if points is None: points = pcd.points_padded()[0]

    print("Rendering...")
    batched_imgs, depth, cameras = render_with_pytorch3d(device, pcd, num_views, points, H, W)

    ndc_grid = get_grid(H, W, device)
    pixel_coords_flat = ndc_grid.reshape(-1, 2) 

    ft_per_point = torch.zeros((len(points), FEATURE_DIMS), dtype=torch.float16, device=device)
    num_hits_per_point = torch.zeros((len(points), 1), dtype=torch.float16, device=device)
    
    for idx in tqdm(range(len(batched_imgs))):
        
        current_depth = depth[idx].to(device).flatten().unsqueeze(1)
        valid_mask = current_depth.squeeze() > 0
        if valid_mask.sum() == 0:
            print("No valid pixels(???)")
            continue
            
        xy_depth_valid = torch.cat((
            pixel_coords_flat[valid_mask], 
            current_depth[valid_mask]
        ), dim=1)

        world_coords = cameras[idx].unproject_points(
            xy_depth_valid, world_coordinates=True, from_ndc=True
        ).to(device)

        # Extract DINO features
        img_rgb = batched_imgs[idx].permute(2, 0, 1).unsqueeze(0).to(device)
        # dino_feat, dino_score, class_idx = get_dino_features_and_score(device, dino_model, img_rgb, score=USE_SCORE)
    
        output_image = ip_controlnet.run_diffusion(ip_pipe, img_rgb, current_depth)
        
        # class_str = imagenet_classes[class_idx]
        # UNCOMMENT TO SAVE VISUALIZATion RENDER
        # pilimg = tpl(img_rgb.squeeze(0))
        pilimg = output_image
        
        
        # pilimg.save(f"renders/RENDER{datetime.now().hour}:{datetime.now().minute}:{datetime.now().second}-{class_str}-{dino_score}.png")        
        pilimg.save(f"diffrender/{datetime.now().hour}:{datetime.now().minute}.png")        
        # exit()
        
    print(f"Time taken: {(time() - t1) / 60:.2f} min")
    
    del batched_imgs, depth, cameras
    torch.cuda.empty_cache()
    
    # return ft_per_point
    
if __name__ == "__main__":

    
    first_FILE = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/sofa/080_00003.bin"
    device = torch.device("cuda")

    first_pcd, first_labels = load_scanobjectnn_to_pytorch3d(first_FILE, device)
    print("computing features for first (pillow a)...")
    get_diffused_depth(device,
                       dino2.init_dino(device),
                       first_pcd,
                       True,
                       ip_controlnet.init_diffusion(),
                       )
    