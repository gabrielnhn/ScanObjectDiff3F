import gc
import torch
import numpy as np
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

FEATURE_DIMS = 768 

def get_corrected_ndc_grid(H, W, device):
    x_range = torch.linspace(1, -1, W, device=device) 
    y_range = torch.linspace(1, -1, H, device=device)
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
    return torch.stack((x_grid, y_grid), dim=-1)

def render_with_pytorch3d(device, pcd, num_views=8, H=512, W=512):
    elev = torch.linspace(0, 0, num_views)
    azim = torch.linspace(0, 360, num_views)
    
    points = pcd.points_padded()[0]
    center = points.mean(0)
    scale = (points - center).abs().max()
    dist = torch.full((num_views,), scale * 2.5) 
    
    R, T = look_at_view_transform(
        dist=dist, elev=elev, azim=azim, at=center.unsqueeze(0).repeat(num_views, 1), device=device
    )
    cameras = PerspectiveCameras(device=device, R=R, T=T)

    raster_settings = PointsRasterizationSettings(
        image_size=(H, W), 
        radius=0.025,
        points_per_pixel=10,
        bin_size=0 
    )

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    
    if pcd.features_padded() is None:
        features = torch.ones_like(points).to(device)
        pcd = pcd.update_features(features)

    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(background_color=(1, 1, 1))
    )

    pcd_batch = pcd.extend(num_views)
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

    print("Rendering...")
    batched_imgs, depth, cameras = render_with_pytorch3d(device, pcd, num_views, H, W)

    ndc_grid = get_corrected_ndc_grid(H, W, device)
    pixel_coords_flat = ndc_grid.reshape(-1, 2) 

    ft_per_point = torch.zeros((len(points), FEATURE_DIMS), dtype=torch.float16, device=device)
    ft_per_point_count = torch.zeros((len(points), 1), dtype=torch.float16, device=device)
    
    print("Extracting features (Chunked NN)...")
    for idx in tqdm(range(len(batched_imgs))):
        
        # 1. Unproject
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

        # 2. Extract DINO
        img_rgb = batched_imgs[idx].permute(2, 0, 1).unsqueeze(0).to(device)
        dino_feat = get_dino_features(device, dino_model, img_rgb)
        
        dino_flat = dino_feat.flatten(2).squeeze(0).T 
        features_valid = dino_flat[valid_mask]

        # 3. Accumulate (Chunked Nearest Neighbor)
        # Process in chunks of 5000 pixels to prevent OOM
        chunk_size = 5000
        num_pixels = world_coords.shape[0]
        
        for i in range(0, num_pixels, chunk_size):
            end = min(i + chunk_size, num_pixels)
            
            # Distance chunk: (Chunk, N_Points)
            dists = torch.cdist(world_coords[i:end], points, p=2)
            closest = torch.argmin(dists, dim=1)
            
            ft_per_point.index_add_(0, closest, features_valid[i:end])
            ft_per_point_count.index_add_(0, closest, torch.ones_like(closest, dtype=torch.float16).unsqueeze(1))
            
            del dists, closest
            
        del world_coords, features_valid, dino_feat, img_rgb, current_depth

    # Average
    mask = (ft_per_point_count > 0).squeeze()
    ft_per_point[mask] /= ft_per_point_count[mask]
    ft_per_point = torch.nan_to_num(ft_per_point)
    
    # Fill remaining holes
    if (~mask).sum() > 0:
        filled_indices = torch.where(mask)[0]
        missing_indices = torch.where(~mask)[0]
        if len(filled_indices) > 0:
            # Chunked fill for missing points
            chunk_size_fill = 5000
            num_missing = len(missing_indices)
            for i in range(0, num_missing, chunk_size_fill):
                end = min(i + chunk_size_fill, num_missing)
                curr_missing = missing_indices[i:end]
                
                dists = torch.cdist(points[curr_missing], points[filled_indices], p=2)
                closest = torch.argmin(dists, dim=1)
                ft_per_point[curr_missing] = ft_per_point[filled_indices][closest]

    print(f"Time taken: {(time() - t1) / 60:.2f} min")
    
    del batched_imgs, depth, cameras
    torch.cuda.empty_cache()
    
    return ft_per_point