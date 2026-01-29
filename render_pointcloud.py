from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor)
import torch
import math
import numpy as np
from pytorch3d.structures import Pointclouds


# --- 2. The Multi-Pass Renderer ---
@torch.no_grad()
def run_rendering(device, pcd, num_views, H, W, add_angle_azi=0, add_angle_ele=0, use_normal_map=False, render_white=False):
    
    bbox = pcd.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bb_diff = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) / 2.0
    scaling_factor = 0.65
    distance = torch.sqrt((bb_diff * bb_diff).sum())
    distance *= scaling_factor
    
    # Grid logic
    steps = int(math.sqrt(num_views))
    if steps < 1: steps = 1
        
    end = 360 - 360/steps
    elevation = torch.linspace(start=0, end=end, steps=steps).repeat(steps) + add_angle_ele
    azimuth = torch.linspace(start=0, end=end, steps=steps)
    azimuth = torch.repeat_interleave(azimuth, steps) + add_angle_azi
    bbox_center = bbox_center.unsqueeze(0)
    
    rotation, translation = look_at_view_transform(
        dist=distance, azim=azimuth, elev=elevation, device=device, at=bbox_center
    )
    camera = PerspectiveCameras(R=rotation, T=translation, device=device)

    # Settings: High radius/points_per_pixel helps make the PC look "solid" for Diff3F
    rasterization_settings = PointsRasterizationSettings(
        image_size=(H, W),
        radius=0.015, 
        points_per_pixel=1, 
        bin_size=0
    )

    rasterizer = PointsRasterizer(cameras=camera, raster_settings=rasterization_settings)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())
    
    actual_views = rotation.shape[0]
    
    # --- PASS 1: RGB (or White) ---
    if render_white:
        # Create a temp PC with white features
        white_feats = torch.ones_like(pcd.points_padded())
        pcd_render = Pointclouds(points=pcd.points_padded(), features=white_feats).to(device)
    else:
        # Use existing RGB features
        pcd_render = pcd
        
    batch_pcd = pcd_render.extend(actual_views)
    batched_renderings = renderer(batch_pcd) # (N, H, W, 4)

    # --- PASS 2: Normals ---
    normal_batched_renderings = None
    if use_normal_map:
        if pcd.normals_padded() is None:
            print("Warning: use_normal_map=True but Point Cloud has no normals!")
        else:
            # 1. Get Normals
            # 2. Map [-1, 1] -> [0, 1] for image representation
            normals_as_color = (pcd.normals_padded() + 1.0) / 2.0
            
            # 3. Create temp PC where features = normals
            pcd_normals = Pointclouds(points=pcd.points_padded(), features=normals_as_color).to(device)
            batch_pcd_normals = pcd_normals.extend(actual_views)
            
            # 4. Render
            normal_batched_renderings = renderer(batch_pcd_normals) # (N, H, W, 4)

    # Get Depth (from the first pass)
    fragments = rasterizer(batch_pcd)
    depth = fragments.zbuf
    
    return batched_renderings, normal_batched_renderings, camera, depth


def batch_render(device, pcd, num_views, H, W, use_normal_map=False):
    trials = 0
    add_angle_azi = 0
    add_angle_ele = 0
    
    render_white = False 
    render_white = True 
    
    while trials < 5:
        try:
            return run_rendering(
                device, pcd, num_views, H, W, 
                add_angle_azi=add_angle_azi, 
                add_angle_ele=add_angle_ele, 
                use_normal_map=use_normal_map,
                render_white=render_white
            )
        except torch.linalg.LinAlgError as e:
            trials += 1
            print("lin alg exception at rendering, retrying ", trials)
            add_angle_azi = torch.randn(1)
            add_angle_ele = torch.randn(1)
            continue