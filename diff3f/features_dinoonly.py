import gc
import torch
import numpy as np
import math
from tqdm import tqdm
from time import time

from pytorch3d.renderer import (
    look_at_view_transform,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor,
    PerspectiveCameras
)
from dino2 import get_dino_features
import torchvision
TPL = torchvision.transforms.ToPILImage
tpl = TPL()




FEATURE_DIMS = 768 

def get_depthonly_style_grid(H, W, device):
    """
    Replicates the 'depthonly' grid logic:
    1. Y is inverted (1 to -1)
    2. X is flipped (1 to -1) to match the mirroring in the renderer
    """
    x_range = torch.linspace(1, -1, W, device=device) # X Flip
    y_range = torch.linspace(1, -1, H, device=device) # Y Invert
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
    return torch.stack((x_grid, y_grid), dim=-1)

def render_with_pytorch3d(device, pcd, num_views, H=512, W=512):
    bbox = pcd.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bb_diff = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) / 2.0
    
    # Distance Logic
    scaling_factor = 0.65
    distance = torch.sqrt((bb_diff * bb_diff).sum())
    distance *= scaling_factor
    
    # Trajectory Logic
    steps = int(math.sqrt(num_views))
    if steps < 1: steps = 1
    end = 360 - 360/steps
    
    # Create grid of views
    elevation = torch.linspace(start=0, end=end, steps=steps).repeat(steps)
    azimuth = torch.linspace(start=0, end=end, steps=steps)
    azimuth = torch.repeat_interleave(azimuth, steps)
    
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
        points_per_pixel=5, 
        bin_size=0 
    )

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    
    # points = pcd.points_padded()[0]
    # features = torch.ones_like(points).to(device)
    # pcd_render = pcd.update_features(features)
    pcd_render = pcd

    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(background_color=(1, 1, 1)) # White Background
    )

    pcd_batch = pcd_render.extend(num_actual_views)
    images = renderer(pcd_batch).cpu()
    fragments = rasterizer(pcd_batch)
    depth = fragments.zbuf[..., 0].cpu()
    
    valid_mask = (fragments.idx[..., 0] != -1).cpu()
    depth[~valid_mask] = -1

    del fragments, pcd_batch
    return images, depth, cameras

def get_features_per_point(
    device, dino_model, pcd, 
    num_views=50, H=512, W=512, tolerance=0.01, 
    points=None, bq=False 
):
    t1 = time()
    if points is None: points = pcd.points_padded()[0]

    print("Rendering (Robust BBox Mode)...")
    batched_imgs, depth, cameras = render_with_pytorch3d(device, pcd, num_views, H, W)

    # Use the grid logic that matches 'depthonly'
    ndc_grid = get_depthonly_style_grid(H, W, device)
    pixel_coords_flat = ndc_grid.reshape(-1, 2) 

    ft_per_point = torch.zeros((len(points), FEATURE_DIMS), dtype=torch.float16, device=device)
    ft_per_point_count = torch.zeros((len(points), 1), dtype=torch.float16, device=device)
    
    print("Extracting features (Chunked NN)...")
    for idx in tqdm(range(len(batched_imgs))):
        
        current_depth = depth[idx].to(device).flatten().unsqueeze(1)
        valid_mask = current_depth.squeeze() > 0
        if valid_mask.sum() == 0: continue
            
        xy_depth_valid = torch.cat((
            pixel_coords_flat[valid_mask], 
            current_depth[valid_mask]
        ), dim=1)

        world_coords = cameras[idx].unproject_points(
            xy_depth_valid, world_coordinates=True, from_ndc=True
        ).to(device)

        # Extract DINO
        img_rgb = batched_imgs[idx].permute(2, 0, 1).unsqueeze(0).to(device)
        
        
        pilimg = tpl(img_rgb.squeeze(0))
        from datetime import datetime
        pilimg.save(f"LATESTRENDER{datetime.now()}.png")        
        
        
        dino_feat = get_dino_features(device, dino_model, img_rgb)
        
        # Flatten
        dino_flat = dino_feat.flatten(2).squeeze(0).T 
        features_valid = dino_flat[valid_mask]

        # Accumulate (Chunked Nearest Neighbor)
        chunk_size = 5000
        num_pixels = world_coords.shape[0]
        
        for i in range(0, num_pixels, chunk_size):
            end = min(i + chunk_size, num_pixels)
            
            # Find closest mesh vertex to this projected pixel
            dists = torch.cdist(world_coords[i:end], points, p=2)
            closest = torch.argmin(dists, dim=1)
            
            ft_per_point.index_add_(0, closest, features_valid[i:end])
            ft_per_point_count.index_add_(0, closest, torch.ones_like(closest, dtype=torch.float16).unsqueeze(1))
            
        del world_coords, features_valid, dino_feat, img_rgb, current_depth

    # Average
    mask = (ft_per_point_count > 0).squeeze()
    ft_per_point[mask] /= ft_per_point_count[mask]
    ft_per_point = torch.nan_to_num(ft_per_point)
    
    # # Fill remaining holes (if any) with nearest valid feature
    # if (~mask).sum() > 0:
    #     filled_indices = torch.where(mask)[0]
    #     missing_indices = torch.where(~mask)[0]
    #     if len(filled_indices) > 0:
    #         print(f"Filling {len(missing_indices)} missing points via NN...")
    #         # Process hole-filling in chunks too
    #         chunk_size_fill = 5000
    #         for i in range(0, len(missing_indices), chunk_size_fill):
    #             end = min(i + chunk_size_fill, len(missing_indices))
    #             curr_missing = missing_indices[i:end]
    #             dists = torch.cdist(points[curr_missing], points[filled_indices], p=2)
    #             closest = torch.argmin(dists, dim=1)
    #             ft_per_point[curr_missing] = ft_per_point[filled_indices][closest]

    print(f"Time taken: {(time() - t1) / 60:.2f} min")
    
    del batched_imgs, depth, cameras
    torch.cuda.empty_cache()
    
    return ft_per_point