import gc
import torch
import numpy as np
import math
from tqdm import tqdm
from time import time

from pc_utils import load_scanobjectnn_to_pytorch3d, save_pointcloud_with_features
import ip_controlnet
import clip
import depth_estimation


CONDITION_SCALE = 0.3
IP_PROMPT_SCALE = 0.25
TEXT_PROMPT = "complete chair, chair, black background, high quality, best quality"
STRENGTH_IMG2IMG = 0.9


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

from pytorch3d.ops import knn_points

def merge_and_color_pointclouds(existing_points, hallucinated_points, threshold=0.03):
    """
    Filters hallucinated points based on distance to existing points.
    Returns combined points and RGB color features for visualization.
    """
    # device = torch.device("cuda")
    
    # Ensure points are 2D tensors [N, 3]
    if existing_points.dim() == 3: existing_points = existing_points.squeeze(0)
    if hallucinated_points.dim() == 3: hallucinated_points = hallucinated_points.squeeze(0)

    # PyTorch3D knn_points expects batched inputs: [Batch, N, 3]
    p1 = hallucinated_points.unsqueeze(0).cuda()
    p2 = existing_points.unsqueeze(0).cuda()

    # Find the single nearest neighbor in existing (p2) for each hallucinated point (p1)
    # Note: knn_points returns SQUARED distances, so we must square our threshold!
    dists, _, _ = knn_points(p1, p2, K=1)
    sq_dists = dists.squeeze(0).squeeze(-1) # Shape: [num_hallucinated]

    # Mask out points that are too close to the existing geometry
    mask = sq_dists > (threshold ** 2)

    hallucinated_points = hallucinated_points.cuda()
    valid_new_points = hallucinated_points[mask]
    
    num_accepted = valid_new_points.shape[0]
    print(f"Accepted {num_accepted} new points out of {hallucinated_points.shape[0]} (Threshold: {threshold})")

    if num_accepted == 0:
        print("No new points passed the distance threshold.")
        valid_new_points = torch.empty((0, 3), device=device)

    # --- COLOR ASSIGNMENT ---
    # Colors in PyTorch3D are usually floats between 0.0 and 1.0
    
    # Existing points: Neutral Light Grey
    existing_colors = torch.full_like(existing_points, 0.7) 
    
    # New points: Bright Green [R=0, G=1, B=0]
    new_colors = torch.zeros_like(valid_new_points)
    new_colors[:, 1] = 1.0 

    # Combine them
    combined_points = torch.cat([existing_points, valid_new_points], dim=0)
    combined_colors = torch.cat([existing_colors, new_colors], dim=0)

    return combined_points, combined_colors


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
    
    renderer = CircleRenderer(background_color=(0,0,0)).to(device)

    pcd_batch = pcd.extend(num_actual_views)

    # do the rendering
    fragments = rasterizer(pcd_batch)
    images = renderer(fragments, pcd_batch).cpu()

    depth = fragments.zbuf[..., 0].cpu()
    valid_mask = (fragments.idx[..., 0] != -1).cpu()
    depth[~valid_mask] = -1
    del fragments, pcd_batch #, pt_features
    return images, depth, cameras

def ransac_depth_alignment(gt_vals, est_vals, num_iters=1000, inlier_thresh=0.05):
    """
    Robustly finds Scale (s) and Shift (t) using RANSAC to ignore flying pixels and edge noise.
    """
    best_inliers = 0
    best_s, best_t = 1.0, 0.0

    if len(gt_vals) < 2:
        return best_s, best_t

    # RANSAC Loop
    for _ in range(num_iters):
        # 1. Randomly sample 2 points to draw a line
        idx = torch.randperm(len(gt_vals), device=gt_vals.device)[:2]
        sample_gt = gt_vals[idx]
        sample_est = est_vals[idx]

        # 2. Solve for s and t: s = (y2 - y1) / (x2 - x1)
        denom = sample_est[0] - sample_est[1]
        if abs(denom) < 1e-6:
            continue
            
        s = (sample_gt[0] - sample_gt[1]) / denom
        t = sample_gt[0] - s * sample_est[0]

        # 3. Calculate errors for ALL points
        pred_gt = est_vals * s + t
        errors = torch.abs(pred_gt - gt_vals)

        # 4. Count inliers (how many pixels agree with this scale/shift?)
        inliers = (errors < inlier_thresh).sum().item()

        if inliers > best_inliers:
            best_inliers = inliers
            best_s = s
            best_t = t
            
    # Polish: Run Least Squares ONLY on the verified inliers to get the absolute best fit
    pred_gt = est_vals * best_s + best_t
    inlier_mask = torch.abs(pred_gt - gt_vals) < inlier_thresh
    
    if inlier_mask.sum() > 2:
        A = torch.vstack([est_vals[inlier_mask], torch.ones_like(est_vals[inlier_mask])]).T
        solution = torch.linalg.lstsq(A, gt_vals[inlier_mask]).solution
        best_s, best_t = solution[0].item(), solution[1].item()

    return best_s, best_t


def get_diffused_depth(
    pcd,
    path_append="",
    index_name=0,
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


    diffused_images = []
    # DIFFUSION STEP 
    for idx in tqdm(range(len(batched_imgs)), desc="Diffusion all POVs"):
        
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
        controlnet_depth_pil.save(os.path.join(renders_dir, f"{idx}a.png"))

        img_rgb = batched_imgs[idx].permute(2, 0, 1)
        current_pov = tpl(img_rgb)
        current_pov.save(os.path.join(renders_dir, f"{idx}b.png"))

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
        output_image.save(os.path.join(renders_dir, f"{idx}d.png"))
        
        diffused_images.append(output_image)
        
    del batched_imgs, cameras, depth_rgb, depth
    del ip_pipe
    
    depther = depth_estimation.init_depther()
        
    for idx in tqdm(range(len(diffused_images)), desc="Extracting all depths"):
        output_image = diffused_images[idx]
        
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
        new_depth_pil.save(os.path.join(renders_dir, f"{idx}e.png"))
        
    print(f"Time taken: {(time() - t1) / 60:.2f} min")
    
    # del batched_imgs, depth, cameras
    torch.cuda.empty_cache()
    

if __name__ == "__main__":
    print("----------")
    print("----------")
    device = torch.device("cuda")

    mvp_file = "/home/gabrielnhn/datasets/MVP_Test_CP.h5"
    
    TEST_INDEX = 9
    
    print(f"Loading Partial Point Cloud index {TEST_INDEX} from {mvp_file}")
    
    from pc_utils import load_mvp_to_pytorch3d 
    
    # Load the incomplete shape to run through your diffusion pipeline
    partial_pcd = load_mvp_to_pytorch3d(
        h5_filename=mvp_file, 
        index=TEST_INDEX, 
        load_complete=False # Give us the broken shape!
    )
    
    gt_pcd = load_mvp_to_pytorch3d(
        h5_filename=mvp_file, 
        index=TEST_INDEX, 
        load_complete=True # Give us the perfect shape!
    )
    save_pointcloud_with_features(gt_pcd, f"GROUND_TRUTH_COMPLETE_SHAPE_{TEST_INDEX}.ply")
    save_pointcloud_with_features(partial_pcd, f"GROUND_TRUTH_PARTIAL_SHAPE_{TEST_INDEX}.ply")

    # Run your pipeline!
    path_name = f"MVP_index_{TEST_INDEX}"
    get_diffused_depth(partial_pcd, path_append=path_name, index_name=TEST_INDEX)