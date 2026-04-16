import torch
import numpy as np
import math
from tqdm import tqdm
from time import time

from pc_utils import load_scanobjectnn_to_pytorch3d, save_pointcloud_with_features
# import ip_controlnet
import zero123_controlnet
import clip
import depth_estimation

# IP_PROMPT_SCALE = 0.25
# STRENGTH_IMG2IMG = 0.8
# RESOLUTION = 512
RESOLUTION = 320
CONDITION_SCALE = 0.2
DEFAULT_TEXT_PROMPT = ""
PROMPT_APPEND_ALWAYS = ", high quality, best quality"

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.renderer import AlphaCompositor

class PhongCircleRenderer(nn.Module):
    """
    Render circles with Blinn-Phong shading. 
    Requires precomputed fragments, packed normals, and camera positions.
    """
    def __init__(self, background_color=(1.0, 1.0, 1.0), 
                 ambient=0.3, diffuse=0.7, specular=0.2, shininess=32.0):
        super().__init__()
        self.compositor = AlphaCompositor(background_color=background_color)
        
        # Phong material properties
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess

    def forward(self, fragments, pcd_batch, cameras=None, light_dir=torch.tensor([0.0, 1.0, 1.0])):
        weights = (fragments.idx != -1).float().permute(0, 3, 1, 2)
        indices = fragments.idx.long().permute(0, 3, 1, 2)
        
        points = pcd_batch.points_packed()
        features = pcd_batch.features_packed()
        normals = pcd_batch.normals_packed()
        
        if normals is None:
            raise ValueError("need normals")

        light_dir = F.normalize(light_dir.to(points.device), p=2, dim=-1)

        n_dot_l = torch.sum(normals * light_dir, dim=-1, keepdim=True)
        diffuse_term = torch.clamp(n_dot_l, min=0.0)

        specular_term = torch.zeros_like(diffuse_term)
        if cameras is not None:
            # Map batch camera centers to the packed points seamlessly
            cloud_idx = pcd_batch.packed_to_cloud_idx()
            cam_centers = cameras.get_camera_center()[cloud_idx]
            
            # Calculate View Direction (V) and Halfway Vector (H)
            view_dir = F.normalize(cam_centers - points, p=2, dim=-1)
            half_vec = F.normalize(light_dir + view_dir, p=2, dim=-1)
            
            n_dot_h = torch.sum(normals * half_vec, dim=-1, keepdim=True)
            specular_term = torch.pow(torch.clamp(n_dot_h, min=0.0), self.shininess)

        # Shaded Color = Base Color * (Ambient + Diffuse) + Specular Highlight
        shaded_features = features * (self.ambient + self.diffuse * diffuse_term) + (self.specular * specular_term)
        shaded_features = torch.clamp(shaded_features, 0.0, 1.0)

        shaded_features = shaded_features.permute(1, 0)

        images = self.compositor(indices, weights, shaded_features)
        return images.permute(0, 2, 3, 1)




def get_grid(H, W, device):
    x_range = torch.linspace(1, -1, W, device=device) # X Flip
    y_range = torch.linspace(1, -1, H, device=device) # Y Invert
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
    return torch.stack((x_grid, y_grid), dim=-1)

from PIL import Image

def create_zero123_depth_grid(depth_pils):
    """
    3x2 depth condition image arrangement
    """
    
    grid = Image.new('RGB', (RESOLUTION*2, RESOLUTION*3))
    positions = [
        (0, 0), (RESOLUTION, 0),
        (0, RESOLUTION), (RESOLUTION, RESOLUTION),
        (0, RESOLUTION*2), (RESOLUTION, RESOLUTION*2)
    ]
    
    for img, pos in zip(depth_pils, positions):
        grid.paste(img, pos)
        
    return grid


def find_best_reference_pov(pcd, device, depth_weight=0.5):
    """
    Adapted from ComPC
    Finds the optimal elevation and azimuth to view a partial point cloud.
    Maximizes visible points (coverage) while minimizing average depth (closeness).
    Optimized for 6GB VRAM using chunked pure-rasterization.
    """
    print("Searching for Canonical Front Face...")
    
    bbox = pcd.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bbox_center = (bbox_min + bbox_max) / 2.0
    distance = torch.sqrt(((bbox_max - bbox_min) ** 2).sum()) * 0.65
    total_points = pcd.points_padded().shape[1]

    # Use a low resolution. We only need point hits, not pretty pictures!
    H, W = 128, 128 
    raster_settings = PointsRasterizationSettings(
        image_size=(H, W), radius=0.03, points_per_pixel=1
    )

    def evaluate_grid(elevations, azimuths, batch_size=16):
        best_score = -float('inf')
        best_elev, best_azim = 0.0, 0.0
        
        # Flatten the grids
        elevs_flat = elevations.flatten()
        azims_flat = azimuths.flatten()
        num_views = len(elevs_flat)
        
        # Chunk to save VRAM
        for i in range(0, num_views, batch_size):
            chunk_elevs = elevs_flat[i:i+batch_size]
            chunk_azims = azims_flat[i:i+batch_size]
            actual_batch_size = len(chunk_elevs)
            
            R, T = look_at_view_transform(
                dist=distance, elev=chunk_elevs, azim=chunk_azims, 
                device=device, at=bbox_center.unsqueeze(0)
            )
            
            cameras = PerspectiveCameras(device=device, R=R, T=T)
            rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
            
            # Extend PCD for the batch and rasterize (NO COLOR RENDERING NEEDED)
            pcd_batch = pcd.extend(actual_batch_size)
            fragments = rasterizer(pcd_batch)
            
            idx_map = fragments.idx[..., 0] # (B, H, W)
            z_map = fragments.zbuf[..., 0]  # (B, H, W)
            
            for b in range(actual_batch_size):
                valid_mask = idx_map[b] != -1
                
                # How many unique points can we actually see?
                visible_points = torch.unique(idx_map[b][valid_mask]).shape[0]
                coverage_ratio = visible_points / total_points
                
                # How close are they? (Normalize by distance so it scales well)
                if visible_points > 0:
                    mean_depth = z_map[b][valid_mask].mean() / distance
                else:
                    mean_depth = torch.tensor(float('inf'))
                
                # Maximize coverage, minimize depth
                score = coverage_ratio - (depth_weight * mean_depth)
                
                if score > best_score:
                    best_score = score
                    best_elev = chunk_elevs[b].item()
                    best_azim = chunk_azims[b].item()
                    
            del pcd_batch, fragments, cameras, rasterizer
            torch.cuda.empty_cache() # Keep the 4050 breathing
            
        return best_elev, best_azim

    # Broad sweep around the object
    coarse_elevs, coarse_azims = torch.meshgrid(
        torch.linspace(-80, 80, 9, device=device),  
        torch.linspace(0, 330, 12, device=device), indexing='ij'
    )
    c_elev, c_azim = evaluate_grid(coarse_elevs, coarse_azims)

    # Drill down around the best coarse angle (+/- 15 degrees)
    fine_elevs, fine_azims = torch.meshgrid(
        torch.linspace(c_elev - 15, c_elev + 15, 7, device=device),
        torch.linspace(c_azim - 15, c_azim + 15, 7, device=device), indexing='ij'
    )
    final_elev, final_azim = evaluate_grid(fine_elevs, fine_azims)

    print(f"Optimal POV Found -> Azimuth: {final_azim:.1f}°, Elevation: {final_elev:.1f}°")
    return final_elev, final_azim

def render_with_pytorch3d(device, pcd, best_elev, best_azim, H=RESOLUTION, W=RESOLUTION):
    bbox = pcd.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bb_diff = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) / 2.0
    
    scaling_factor = 0.65
    distance = torch.sqrt((bb_diff * bb_diff).sum())
    distance *= scaling_factor
    
    # --- ZERO123++ OFFSET LOGIC ---
    # View 0 is our perfect Reference Image (no offset)
    # Views 1-6 are the specific Zero123++ grid offsets
    zero123_offsets = [
        (0, 0),         # Reference Image (Index 0)
        (30, -20),      # Grid Top-Left (Index 1)
        (90, -20),      # Grid Top-Right (Index 2)
        (150, -20),     # Grid Mid-Left (Index 3)
        (210, 20),      # Grid Mid-Right (Index 4)
        (270, 20),      # Grid Bot-Left (Index 5)
        (330, 20)       # Grid Bot-Right (Index 6)
    ]
    
    # Calculate final absolute angles by adding the baseline to the offsets
    azimuths = [best_azim + az_off for az_off, el_off in zero123_offsets]
    elevations = [best_elev + el_off for az_off, el_off in zero123_offsets]
    
    azimuths_tensor = torch.tensor(azimuths, dtype=torch.float32, device=device)
    elevations_tensor = torch.tensor(elevations, dtype=torch.float32, device=device)
    
    # Calculate R, T for all 7 cameras simultaneously
    R, T = look_at_view_transform(
        dist=distance, 
        elev=elevations_tensor, 
        azim=azimuths_tensor, 
        device=device, 
        at=bbox_center.unsqueeze(0)
    )
    
    cameras = PerspectiveCameras(device=device, R=R, T=T)
    num_actual_views = R.shape[0] # Will be exactly 7
    
    raster_settings = PointsRasterizationSettings(
        image_size=(H, W), 
        radius=0.02,        
        points_per_pixel=1, 
        bin_size=0 
    )
    
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    
    renderer = PhongCircleRenderer(
        # background_color=(0,0,0)
        background_color=(1,1,1)
                              ).to(device)

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
    text_prompt = None,
    num_views=50, H=RESOLUTION, W=RESOLUTION, 
):
    renders_dir = os.path.join("renders",
    f"usingZero123"+path_append)
    
    if not os.path.isdir(renders_dir):
        os.mkdir(renders_dir)    
    
    if text_prompt is None:
        text_prompt = DEFAULT_TEXT_PROMPT
    
    text_prompt += PROMPT_APPEND_ALWAYS
    
    t1 = time()
    # if points is None: 
    points = pcd.points_padded()[0]

    # SELECT BEST POINT OF VIEW!!!
    # semanticity_model, semanticity_processor = clip.init_clip()

    # best_pov_idx = -1
    # best_dino_score = 0
    # for idx in tqdm(range(len(batched_imgs)), desc="Finding best POV"):
    #     img_rgb = batched_imgs[idx].permute(2, 0, 1).unsqueeze(0).to(device)
    #     # dino_feat, dino_score, class_idx = get_dino_features_and_score(device, semanticity_model, img_rgb, score=USE_SCORE)
    #     # del dino_feat, img_rgb, class_idx
    #     semanticity_score = clip.clip_score(
    #         semanticity_model, semanticity_processor, img_rgb,
    #         prompt="horse, outline of a horse"+PROMPT_APPEND_ALWAYS)
        
    #     if semanticity_score > best_dino_score:
    #         best_dino_score = semanticity_score
    #         best_pov_idx = idx
            
    #     del semanticity_score
    
    # best_pov_image = batched_imgs[best_pov_idx].permute(2, 0, 1)
    # best_pov_image = tpl(best_pov_image)
    # del semanticity_model, semanticity_processor
    
    print("Best PoV according to filter on partial renders...")
    best_elev, best_azim = find_best_reference_pov(pcd, device)

    print("Actual Rendering...")
    batched_imgs, depth, cameras = render_with_pytorch3d(device, pcd, best_elev, best_azim)
    best_pov_image = batched_imgs[0].permute(2, 0, 1)
    best_pov_image = tpl(best_pov_image)
    best_pov_image.save(os.path.join(renders_dir, "REFERENCE.png"))

    ndc_grid = get_grid(H, W, device)
    pixel_coords_flat = ndc_grid.reshape(-1, 2) 
    
    depth_images = []
    # arrange depths in grid
    for idx in tqdm(range(len(batched_imgs)), desc="Computing Depth"):
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

        # INVERT THE WHOLE THING DONT CARE
        depth_norm = 1 - depth_norm

        depth_uint8 = (depth_norm * 255).cpu().numpy().astype(np.uint8)
        depth_rgb = np.stack([depth_uint8] * 3, axis=-1)
        
        depth_pil = Image.fromarray(depth_rgb)
        now = str(datetime.now()).replace(":", "-").replace(".", "-")
        # img_rgb = batched_imgs[idx].permute(2, 0, 1)
        # current_pov = tpl(img_rgb)
        # current_pov.save(os.path.join(renders_dir, f"{idx}b.png"))
        
        depth_pil.save(os.path.join(renders_dir, f"{idx}a.png"))
        depth_images.append(depth_pil)


    depth_grid = create_zero123_depth_grid(depth_images)
    depth_grid.save(os.path.join(renders_dir, f"DEPTHGRID.png"))
    
    zero123_pipe = zero123_controlnet.init_diffusion(
        conditioning_scale=CONDITION_SCALE
    )

    diffused_images = []
    # DIFFUSION STEP 
        
    output_image = zero123_controlnet.run_diffusion(
        zero123_pipe,        
        best_pov_image,       
        depth_grid,
        text_prompt=text_prompt,
    )
    # output_image.save(f"diffrender/{now}d.png")
    output_image.save(os.path.join(renders_dir, f"OUTPUTGRID.png"))
    
    del batched_imgs, cameras, depth_rgb, depth
    del zero123_pipe
    del depth_images
    
    
    depther = depth_estimation.init_depther()
        
    # for idx in tqdm(range(len(diffused_images)), desc="Extracting all depths"):
    # output_image = diffused_images[idx]
    
    new_depth_tensor = depth_estimation.get_depth_map(depther, output_image)
    min_d = new_depth_tensor.min()
    max_d = new_depth_tensor.max()
    if max_d > min_d:
        new_depth_norm = (new_depth_tensor - min_d) / (max_d - min_d)
        new_depth_norm = 1.0 - new_depth_norm
    else:
        new_depth_norm = torch.ones_like(new_depth_tensor)
        
    new_depth_uint8 = (new_depth_norm * 255).byte()
    new_depth_pil = tpl(new_depth_uint8)
    new_depth_pil.save(os.path.join(renders_dir, f"REEXTRACTED_DEPTH.png"))
        
    print(f"Time taken: {(time() - t1) / 60:.2f} min")
    
    # del batched_imgs, depth, cameras
    torch.cuda.empty_cache()
    
if __name__ == "__main__":
    print("----------")
    print("----------")
    device = torch.device("cuda")
    dataset_path = "/home/gabrielnhn/datasets/synthetic_redwood/upload/plyobj"    
    object = "horse.ply"
    
    from pc_utils import load_ply_to_pytorch3d 
    # Load the incomplete shape to run through your diffusion pipeline
    partial_pcd = load_ply_to_pytorch3d(os.path.join(dataset_path, "indata", object))
    gt_pcd = load_ply_to_pytorch3d(os.path.join(dataset_path, "gtdata", object))
    
    save_pointcloud_with_features(gt_pcd, f"GROUND_TRUTH_COMPLETE_SHAPE.ply")
    save_pointcloud_with_features(partial_pcd, f"GROUND_TRUTH_PARTIAL_SHAPE.ply")
    path_name = f"RedWood"
    get_diffused_depth(partial_pcd, path_append=path_name,
        text_prompt="horse, complete horse, black background"
    )
