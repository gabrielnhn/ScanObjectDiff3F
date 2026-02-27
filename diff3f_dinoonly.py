import gc
import torch
import numpy as np
from tqdm import tqdm
from time import time
import random

# PyTorch3D imports for the standard renderer
from pytorch3d.renderer import (
    look_at_view_transform,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor,
    PerspectiveCameras
)
from pytorch3d.ops import ball_query

# Imports from your project structure
# from dino import get_dino_features
from dino2 import get_dino_features

# --- CONFIG ---
FEATURE_DIMS = 768 # DINOv2-VitB usually 768 (Adjust based on your specific DINO model)
VERTEX_GPU_LIMIT = 35000

import torchvision
to_pil_image = torchvision.transforms.ToPILImage()


def arange_pixels(resolution=(128, 128), batch_size=1, subsample_to=None, invert_y_axis=False, margin=0, corner_aligned=True, jitter=None):
    h, w = resolution
    n_points = resolution[0] * resolution[1]
    uh = 1 if corner_aligned else 1 - (1 / h)
    uw = 1 if corner_aligned else 1 - (1 / w)
    if margin > 0:
        uh = uh + (2 / h) * margin
        uw = uw + (2 / w) * margin
        w, h = w + margin * 2, h + margin * 2

    x, y = torch.linspace(-uw, uw, w), torch.linspace(-uh, uh, h)
    if jitter is not None:
        dx = (torch.ones_like(x).uniform_() - 0.5) * 2 / w * jitter
        dy = (torch.ones_like(y).uniform_() - 0.5) * 2 / h * jitter
        x, y = x + dx, y + dy
    x, y = torch.meshgrid(x, y, indexing='xy') # Explicit indexing
    pixel_scaled = torch.stack([x, y], -1).reshape(1, -1, 2).repeat(batch_size, 1, 1)

    if invert_y_axis:
        pixel_scaled[..., -1] *= -1.0

    return pixel_scaled.to(device="cuda")

def render_with_pytorch3d(device, pcd, num_views=8, H=512, W=512):
    """
    Standard PyTorch3D Point Cloud Renderer.
    Generates a circular trajectory and renders the PC.
    """
    # 1. Generate circular camera trajectory
    elev = torch.linspace(0, 0, num_views) # Elevation 0
    azim = torch.linspace(0, 360, num_views) # Full circle
    
    # Distance: heuristics based on point cloud bounding box
    points = pcd.points_padded()[0]
    center = points.mean(0)
    scale = (points - center).abs().max()
    dist = torch.full((num_views,), scale * 2.5) 
    
    R, T = look_at_view_transform(
        dist=dist, 
        elev=elev, 
        azim=azim, 
        at=center.unsqueeze(0).repeat(num_views, 1), 
        device=device
    )
    
    cameras = PerspectiveCameras(device=device, R=R, T=T)

    # 2. Setup Rasterizer and Renderer
    raster_settings = PointsRasterizationSettings(
        image_size=(H, W), 
        radius=0.015,
        points_per_pixel=10,
        bin_size=0  # <--- THE FIX: 0 enables naive rasterization (prevents overflow)
    )

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    
    # If pcd has no features (colors), give it white features
    if pcd.features_padded() is None:
        features = torch.ones_like(points).to(device)
        pcd_render = pcd.update_features(features)
    else:
        pcd_render = pcd

    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(
                # background_color=(0, 0, 0)
                background_color=(1, 1, 1)
            )
    )

    # 3. Render
    pcd_batch = pcd_render.extend(num_views)
    
    # images = renderer(pcd_batch)
    images = renderer(pcd_batch).cpu()
    
    fragments = rasterizer(pcd_batch)
    # depth = fragments.zbuf[..., 0] 
    depth = fragments.zbuf[..., 0].cpu()
    
    valid_mask = fragments.idx[..., 0] != -1
    valid_mask = valid_mask.cpu()
    depth[~valid_mask] = -1

    del fragments,pcd_batch

    return images, depth, cameras

def get_features_per_point(
    device,
    dino_model,
    pcd,
    num_views=100,
    H=512,
    W=512,
    tolerance=0.01,
    points=None,
    bq=True,
):
    t1 = time()
    
    if points is None:
        points = pcd.points_padded()[0]

    # --- MEMORY OPTIMIZATION FOR BALL QUERY RADIUS ---
    points_cpu = points.cpu()
    if len(points_cpu) > VERTEX_GPU_LIMIT:
        samples = random.sample(range(len(points_cpu)), 10000)
        maximal_distance = torch.cdist(points_cpu[samples], points_cpu[samples]).max()
    else:
        maximal_distance = torch.cdist(points_cpu, points_cpu).max()
    del points_cpu
    ball_drop_radius = maximal_distance * tolerance
    # ---------------------------

    print("Rendering with standard PyTorch3D...")
    batched_imgs, depth, cameras = render_with_pytorch3d(device, pcd, num_views, H, W)

    # Grid setup
    pixel_coords = arange_pixels((H, W), invert_y_axis=True)[0]
    # pixel_coords[:, 0] = torch.flip(pixel_coords[:, 0], dims=[0])
    grid = arange_pixels((H, W), invert_y_axis=False)[0].to(device).reshape(1, H, W, 2).half()
    
    torch.cuda.empty_cache()
    gc.collect()
    
    ft_per_point = torch.zeros((len(points), FEATURE_DIMS)).to(device).half()
    ft_per_point_count = torch.zeros((len(points), 1)).to(device).half()
    
    print("Extracting DINO features...")
    
    num_actual_views = len(batched_imgs)
    
    for idx in tqdm(range(num_actual_views)):
        
        # 1. Prepare Depth
        current_depth_map = depth[idx].to(device)
        dp = current_depth_map.flatten().unsqueeze(1) 
        xy_depth = torch.cat((pixel_coords, dp), dim=1) 
        indices = xy_depth[:, 2] > 0 
        xy_depth = xy_depth[indices]
        
        if len(xy_depth) == 0:
            continue

        world_coords = (
            cameras[idx].unproject_points(
                xy_depth, world_coordinates=True, from_ndc=True
            )
        ).to(device)

        # 2. Extract DINO
        img_rgb = batched_imgs[idx].permute(2, 0, 1).unsqueeze(0).to(device) 
        
        # img_pil = to_pil_image(img_rgb.squeeze(0))
        # img_pil.save(f"render{idx}.png")            
        
        dino_features = get_dino_features(device, dino_model, img_rgb, grid)
        
        with torch.no_grad():
            aligned_features = torch.nn.functional.normalize(dino_features, dim=1)

        if aligned_features.dim() == 4:
            aligned_features = aligned_features.flatten(2)
            
        features_per_pixel = aligned_features[0, :, indices].T # (N_pixels, Dim)

        # 3. Map to 3D Points
        if bq:
            # queried_indices is (N_pixels, K=50)
            queried_indices = (
                ball_query(
                    world_coords.unsqueeze(0),
                    points.unsqueeze(0),
                    K=50,
                    radius=ball_drop_radius,
                    return_nn=False,
                )
                .idx[0]
                .to(device)
            )
            
            # --- THE FIX: LOOP OVER NEIGHBORS (K) INSTEAD OF EXPANDING ---
            K = queried_indices.shape[1] # 50
            
            for k in range(K):
                # Get the k-th neighbor for all pixels
                neighbor_indices = queried_indices[:, k] # (N_pixels,)
                
                # Filter valid neighbors (-1 indicates no neighbor found)
                valid_mask = neighbor_indices != -1
                
                # Select only the valid indices and corresponding features
                valid_neighbor_indices = neighbor_indices[valid_mask]
                valid_features = features_per_pixel[valid_mask] # Size: (~N_pixels, Dim) -> ~30MB
                
                # Accumulate
                ft_per_point.index_add_(0, valid_neighbor_indices, valid_features)
                ft_per_point_count.index_add_(0, valid_neighbor_indices, torch.ones_like(valid_neighbor_indices, dtype=torch.float16).unsqueeze(1))
            
            # -----------------------------------------------------------

        else:
            distances = torch.cdist(world_coords, points, p=2)
            closest_vertex_indices = torch.argmin(distances, dim=1)
            
            ft_per_point.index_add_(0, closest_vertex_indices, features_per_pixel.half())
            ft_per_point_count.index_add_(0, closest_vertex_indices, torch.ones_like(closest_vertex_indices, dtype=torch.float16).unsqueeze(1))

        # Cleanup
        del world_coords, aligned_features, features_per_pixel, dino_features, img_rgb, current_depth_map, queried_indices
    
    # Average features
    mask_count = (ft_per_point_count > 0).squeeze()
    ft_per_point[mask_count] = ft_per_point[mask_count] / ft_per_point_count[mask_count]
    
    missing_features = (~mask_count).sum().item()
    print(f"Number of points with no features: {missing_features}")

    t2 = time() - t1
    print(f"Time taken in mins: {t2 / 60:.2f}")
    
    del batched_imgs, depth, cameras
    torch.cuda.empty_cache()
    
    return ft_per_point