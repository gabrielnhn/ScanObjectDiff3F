import gc
import torch
import numpy as np
import math
from tqdm import tqdm
from time import time

from pc_utils import load_scanobjectnn_to_pytorch3d
import ip_controlnet
import clip


CONDITION_SCALE = 0.4
IP_PROMPT_SCALE = 0.75
TEXT_PROMPT = "back of sofa, back of couch, white background, high quality, best quality"
STRENGTH_IMG2IMG = 0.6


from pytorch3d.renderer import (
    look_at_view_transform,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras
)
import torchvision
TPL = torchvision.transforms.ToPILImage
tpl = TPL()


device = torch.device("cuda")

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
    pcd,
    path_append="",
    num_views=50, H=512, W=512, 
):
    renders_dir = os.path.join("renders",
    f"{path_append}im2im cond-{CONDITION_SCALE} ip-{IP_PROMPT_SCALE} str-{STRENGTH_IMG2IMG} pr-{TEXT_PROMPT.replace(',', '')}")
    
    if not os.path.isdir(renders_dir):
        os.mkdir(renders_dir)    
    
    
    t1 = time()
    # if points is None: 
    points = pcd.points_padded()[0]

    print("Rendering...")
    batched_imgs, depth, cameras = render_with_pytorch3d(device, pcd, num_views, points, H, W)

    ndc_grid = get_grid(H, W, device)
    pixel_coords_flat = ndc_grid.reshape(-1, 2) 

    # SELECT BEST POINT OF VIEW!!!
    semanticity_model, semanticity_processor = clip.init_clip()

    best_pov_idx = -1
    best_dino_score = 0
    for idx in tqdm(range(len(batched_imgs)), desc="Finding best POV"):
        img_rgb = batched_imgs[idx].permute(2, 0, 1).unsqueeze(0).to(device)
        # dino_feat, dino_score, class_idx = get_dino_features_and_score(device, semanticity_model, img_rgb, score=USE_SCORE)
        # del dino_feat, img_rgb, class_idx
        semanticity_score = clip.clip_score(semanticity_model, semanticity_processor, img_rgb)
        
        if semanticity_score > best_dino_score:
            best_dino_score = semanticity_score
            best_pov_idx = idx
            
        del semanticity_score
    
    best_pov_image = batched_imgs[best_pov_idx].permute(2, 0, 1)
    best_pov_image = tpl(best_pov_image)
    del semanticity_model, semanticity_processor

    # best_pov_image.save(f"diffrender/REFERENCE.png")
    best_pov_image.save(os.path.join(renders_dir, "REFERENCE.png"))
    
    ip_pipe = ip_controlnet.init_diffusion()

    # DIFFUSION STEP 
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

        depth_2d = depth[idx].clone() # Shape: (H, W)
        valid_mask_2d = depth_2d > 0
        
        if valid_mask_2d.sum() > 0:
            min_depth = depth_2d[valid_mask_2d].min()
            max_depth = depth_2d[valid_mask_2d].max()
            
            if max_depth > min_depth:
                # Normalize to [0, 1]
                depth_norm = (depth_2d - min_depth) / (max_depth - min_depth)
                # Invert for MiDaS style: Closer = 1.0 (White), Farther = 0.0 (Black)
                depth_norm = 1.0 - depth_norm
            else:
                depth_norm = torch.ones_like(depth_2d)
                
            depth_norm[~valid_mask_2d] = 0.0
        else:
            depth_norm = torch.zeros_like(depth_2d)

        depth_uint8 = (depth_norm * 255).cpu().numpy().astype(np.uint8)
        depth_rgb = np.stack([depth_uint8] * 3, axis=-1)
        
        from PIL import Image
        controlnet_depth_pil = Image.fromarray(depth_rgb)
        now = str(datetime.now()).replace(":", "-").replace(".", "-")
        # controlnet_depth_pil.save(f"diffrender/{now}a.png")
        controlnet_depth_pil.save(os.path.join(renders_dir, f"{now}a.png"))

        img_rgb = batched_imgs[idx].permute(2, 0, 1)
        current_pov = tpl(img_rgb)
        current_pov.save(os.path.join(renders_dir, f"{now}b.png"))

        # RUN DIFFUSION
        output_image = ip_controlnet.run_diffusion(
            ip_pipe,        
            best_pov_image,       
            controlnet_depth_pil,
            current_pov,
            condition_scale=CONDITION_SCALE,
            ip_prompt_scale=IP_PROMPT_SCALE,
            text_prompt=TEXT_PROMPT,
            strength=STRENGTH_IMG2IMG
        )
        
        # output_image.save(f"diffrender/{now}d.png")
        output_image.save(os.path.join(renders_dir, f"{now}d.png"))
        
        
    print(f"Time taken: {(time() - t1) / 60:.2f} min")
    
    del batched_imgs, depth, cameras
    torch.cuda.empty_cache()
    
    
if __name__ == "__main__":
    print("----------")
    print("----------")
    print("----------")
    device = torch.device("cuda")

    # first_FILE = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/sofa/080_00003.bin"
    pcd_file ="/home/gabrielnhn/datasets/object_dataset_complete_with_parts/sofa/294_00002.bin"
    
    first_pcd, first_labels = load_scanobjectnn_to_pytorch3d(pcd_file, device)
    print(f"Processing {pcd_file}")
    get_diffused_depth(first_pcd, os.path.basename(pcd_file).split(".")[0])
    
