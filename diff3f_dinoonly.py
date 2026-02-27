import gc
import torch
import numpy as np
from tqdm import tqdm
from time import time
import random

# PyTorch3D imports
from pytorch3d.renderer import (
    look_at_view_transform,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor,
    PerspectiveCameras
)
from pytorch3d.ops import ball_query
from dino2 import get_dino_features

# --- CONFIG ---
FEATURE_DIMS = 768 
VERTEX_GPU_LIMIT = 35000

def get_ndc_coordinates(H, W, device):
    """
    Generates NDC coordinates matching PyTorch3D convention.
    (-1, -1) is Top-Left? No, PyTorch3D is:
    +X = Right
    +Y = Up (So top of image is +1, bottom is -1)
    """
    # Linspace for X (Left to Right: -1 to 1)
    x_range = torch.linspace(-1, 1, W, device=device)
    # Linspace for Y (Top to Bottom: 1 to -1) <--- CRITICAL: Y axis is inverted in NDC relative to pixel row index
    y_range = torch.linspace(1, -1, H, device=device) 
    
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
    
    # Stack to (H, W, 2)
    return torch.stack((x_grid, y_grid), dim=-1)

def render_with_pytorch3d(device, pcd, num_views=8, H=512, W=512):
    # ... (Same camera/rasterizer setup as before) ...
    # 1. Generate circular camera trajectory
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

    # 2. Setup Rasterizer and Renderer
    raster_settings = PointsRasterizationSettings(
        image_size=(H, W), 
        radius=0.02, # Increased radius slightly for better coverage
        points_per_pixel=10,
        bin_size=0 
    )

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    
    if pcd.features_padded() is None:
        features = torch.ones_like(points).to(device)
        pcd_render = pcd.update_features(features)
    else:
        pcd_render = pcd

    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(background_color=(1, 1, 1))
    )

    # 3. Render
    pcd_batch = pcd_render.extend(num_views)
    images = renderer(pcd_batch).cpu()
    fragments = rasterizer(pcd_batch)
    depth = fragments.zbuf[..., 0].cpu() # (N, H, W)
    
    # Fix valid mask logic
    valid_mask = (fragments.idx[..., 0] != -1).cpu()
    depth[~valid_mask] = -1

    del fragments, pcd_batch
    return images, depth, cameras

def get_features_per_point(device, dino_model, pcd, num_views=100, H=512, W=512, tolerance=0.01, points=None, bq=True):
    t1 = time()
    if points is None:
        points = pcd.points_padded()[0]

    # Radius Calculation
    points_cpu = points.cpu()
    if len(points_cpu) > VERTEX_GPU_LIMIT:
        samples = random.sample(range(len(points_cpu)), 10000)
        maximal_distance = torch.cdist(points_cpu[samples], points_cpu[samples]).max()
    else:
        maximal_distance = torch.cdist(points_cpu, points_cpu).max()
    del points_cpu
    ball_drop_radius = maximal_distance * tolerance

    print("Rendering...")
    batched_imgs, depth, cameras = render_with_pytorch3d(device, pcd, num_views, H, W)

    # --- COORDINATE FIX ---
    # Use standard NDC grid. No more manual flips or custom arange.
    pixel_coords = get_ndc_coordinates(H, W, device) # (H, W, 2)
    pixel_coords = pixel_coords.reshape(-1, 2) # (H*W, 2)
    
    ft_per_point = torch.zeros((len(points), FEATURE_DIMS)).to(device).half()
    ft_per_point_count = torch.zeros((len(points), 1)).to(device).half()
    
    print("Extracting features...")
    
    for idx in tqdm(range(len(batched_imgs))):
        
        # 1. Unprojection
        current_depth_map = depth[idx].to(device).flatten() # (H*W)
        
        # Filter valid pixels (depth > 0)
        # Assuming background is -1. Check your render output if background is 0 or -1.
        valid_indices = current_depth_map > 0
        
        if valid_indices.sum() == 0:
            continue
            
        # Select valid coords and depth
        valid_pixel_coords = pixel_coords[valid_indices] # (N_valid, 2)
        valid_depth = current_depth_map[valid_indices].unsqueeze(1) # (N_valid, 1)
        
        xy_depth = torch.cat((valid_pixel_coords, valid_depth), dim=1) # (N_valid, 3)

        # Unproject to World
        world_coords = cameras[idx].unproject_points(
            xy_depth, world_coordinates=True, from_ndc=True
        ).to(device)

        # 2. Extract DINO Features
        img_rgb = batched_imgs[idx].permute(2, 0, 1).unsqueeze(0).to(device) # (1, 3, H, W)
        
        # New DINO function returns (1, Dim, H, W)
        dino_features = get_dino_features(device, dino_model, img_rgb) 
        
        # Flatten to (1, Dim, H*W)
        dino_flat = dino_features.flatten(2) 
        
        # Select features for valid pixels
        features_per_pixel = dino_flat[0, :, valid_indices].T # (N_valid, Dim)

        # 3. Accumulate to 3D Points
        if bq:
            queried_indices = ball_query(
                world_coords.unsqueeze(0),
                points.unsqueeze(0),
                K=50,
                radius=ball_drop_radius,
                return_nn=False
            ).idx[0].to(device)
            
            K = queried_indices.shape[1]
            for k in range(K):
                neighbor_indices = queried_indices[:, k]
                valid_mask = neighbor_indices != -1
                
                valid_neighbor_indices = neighbor_indices[valid_mask]
                valid_features = features_per_pixel[valid_mask]
                
                ft_per_point.index_add_(0, valid_neighbor_indices, valid_features)
                ft_per_point_count.index_add_(0, valid_neighbor_indices, torch.ones_like(valid_neighbor_indices, dtype=torch.float16).unsqueeze(1))
        else:
            distances = torch.cdist(world_coords, points, p=2)
            closest_vertex_indices = torch.argmin(distances, dim=1)
            ft_per_point.index_add_(0, closest_vertex_indices, features_per_pixel.half())
            ft_per_point_count.index_add_(0, closest_vertex_indices, torch.ones_like(closest_vertex_indices, dtype=torch.float16).unsqueeze(1))

        del world_coords, features_per_pixel, dino_features, img_rgb, current_depth_map

    # Average features
    mask_count = (ft_per_point_count > 0).squeeze()
    ft_per_point[mask_count] = ft_per_point[mask_count] / ft_per_point_count[mask_count]
    
    missing_features = (~mask_count).sum().item()
    print(f"Number of points with no features: {missing_features}")
    
    del batched_imgs, depth, cameras
    torch.cuda.empty_cache()
    
    return ft_per_point