import torch
from PIL import Image
from torchvision.utils import make_grid
import numpy as np
from diffusion import add_texture_to_render
from dino import get_dino_features
# Import your new PC renderer
from render_pointcloud import batch_render as pc_batch_render 
from pytorch3d.ops import ball_query
from tqdm import tqdm
from time import time
import random
import gc 
import cv2

FEATURE_DIMS = 1280 + 768  # diffusion unet + dino
VERTEX_GPU_LIMIT = 35000

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
    x, y = torch.meshgrid(x, y)
    pixel_scaled = torch.stack([x, y], -1).permute(1, 0, 2).reshape(1, -1, 2).repeat(batch_size, 1, 1)

    if subsample_to is not None and subsample_to > 0 and subsample_to < n_points:
        idx = np.random.choice(pixel_scaled.shape[1], size=(subsample_to,), replace=False)
        pixel_scaled = pixel_scaled[:, idx]

    if invert_y_axis:
        pixel_scaled[..., -1] *= -1.0

    return pixel_scaled


def get_features_per_point(
    device,
    pipe,
    dino_model,
    pcd,
    prompt,
    num_views=100,
    H=512,
    W=512,
    tolerance=0.01,
    use_latent=False,
    use_normal_map=False, 
    num_images_per_prompt=1,
    points=None,
    return_image=True,
    bq=True,
    prompts_list=None,
):
    t1 = time()
    
    # Extract points from PyTorch3D structure if not provided
    if points is None:
        points = pcd.points_padded()[0] 

    # --- MEMORY OPTIMIZATION ---
    points_cpu = points.cpu()
    if len(points_cpu) > VERTEX_GPU_LIMIT:
        samples = random.sample(range(len(points_cpu)), 10000)
        maximal_distance = torch.cdist(points_cpu[samples], points_cpu[samples]).max()
    else:
        maximal_distance = torch.cdist(points_cpu, points_cpu).max()
    del points_cpu
    # ---------------------------

    ball_drop_radius = maximal_distance * tolerance
    
    # Call the PC Renderer
    batched_renderings, normal_batched_renderings, camera, depth = pc_batch_render(
        device, pcd, num_views, H, W, use_normal_map
    )
    
    print("Rendering complete")
    
    # Move tensors to CPU
    if use_normal_map and normal_batched_renderings is not None:
        normal_batched_renderings = normal_batched_renderings.cpu()
    
    batched_renderings = batched_renderings.cpu()
    camera = camera.cpu()
    depth = depth.cpu()
    
    # Grid setup
    pixel_coords = arange_pixels((H, W), invert_y_axis=True)[0]
    pixel_coords[:, 0] = torch.flip(pixel_coords[:, 0], dims=[0])
    grid = arange_pixels((H, W), invert_y_axis=False)[0].to(device).reshape(1, H, W, 2).half()
    
    torch.cuda.empty_cache()
    
    ft_per_point = torch.zeros((len(points), FEATURE_DIMS)).half()
    ft_per_point_count = torch.zeros((len(points), 1)).half()
    
    print("Extracting features from views...")
    for idx in tqdm(range(len(batched_renderings))):
        current_depth_map = depth[idx, :, :, 0]
        
        dp = current_depth_map.flatten().unsqueeze(1)
        xy_depth = torch.cat((pixel_coords, dp), dim=1)
        
        # Filter background (where depth is -1 or invalid)
        indices = xy_depth[:, 2] != -1 
        xy_depth = xy_depth[indices]
        
        world_coords = (
            camera[idx].unproject_points(
                xy_depth, world_coordinates=True, from_ndc=True
            )
        ).to(device)

        img_rgb = batched_renderings[idx, :, :, :3]
        diffusion_input_img = (img_rgb.cpu().numpy() * 255).astype(np.uint8)
        
        
        # cv2.imshow("RENDER", diffusion_input_img)
        # cv2.waitKey(800)
        
        
        
        normal_map_input = None
        if use_normal_map and normal_batched_renderings is not None:
            normal_map_input = normal_batched_renderings[idx][:, :, :3]
            
        # We reuse the sliced depth map we created earlier
        depth_map = current_depth_map.unsqueeze(0).to(device)
        # cv2.imshow("DEPTH", depth_map)
        # cv2.waitKey(800)
        
        if prompts_list is not None:
            prompt = random.choice(prompts_list)
            
        diffusion_output = add_texture_to_render(
            pipe,
            diffusion_input_img,
            depth_map,
            prompt,
            normal_map_input=normal_map_input,
            use_latent=use_latent,
            num_images_per_prompt=num_images_per_prompt,
            return_image=return_image
        )
        
        aligned_dino_features = get_dino_features(device, dino_model, diffusion_output[1][0], grid)
        
        with torch.no_grad():
            ft = torch.nn.Upsample(size=(H,W), mode="bilinear")(diffusion_output[0].unsqueeze(0)).to(device)
            ft_dim = ft.size(1)
            aligned_features = torch.nn.functional.grid_sample(
                ft, grid, align_corners=False
            ).reshape(1, ft_dim, -1)
            aligned_features = torch.nn.functional.normalize(aligned_features, dim=1)
            
        aligned_features = torch.hstack([aligned_features*0.5, aligned_dino_features*0.5])
        features_per_pixel = aligned_features[0, :, indices].cpu()
        
        if bq:
            queried_indices = (
                ball_query(
                    world_coords.unsqueeze(0),
                    points.unsqueeze(0),
                    K=100,
                    radius=ball_drop_radius,
                    return_nn=False,
                )
                .idx[0]
                .cpu()
            )
            mask = queried_indices != -1
            repeat = mask.sum(dim=1)
            
            # Accumulate features
            ft_per_point_count[queried_indices[mask]] += 1
            ft_per_point[queried_indices[mask]] += features_per_pixel.repeat_interleave(
                repeat, dim=1
            ).T
        else:
            distances = torch.cdist(world_coords, points, p=2)
            closest_vertex_indices = torch.argmin(distances, dim=1).cpu()
            ft_per_point[closest_vertex_indices] += features_per_pixel.T
            ft_per_point_count[closest_vertex_indices] += 1
            
        # Cleanup loop variables
        del diffusion_output, aligned_features, aligned_dino_features, world_coords

    # Average features
    idxs = (ft_per_point_count != 0)[:, 0]
    ft_per_point[idxs, :] = ft_per_point[idxs, :] / ft_per_point_count[idxs, :]
    
    missing_features = len(ft_per_point_count[ft_per_point_count == 0])
    print(f"Number of missing features: {missing_features}")

    # Fill missing points using nearest neighbors
    if missing_features > 0:
        filled_indices = ft_per_point_count[:, 0] != 0
        missing_indices = ft_per_point_count[:, 0] == 0
        
        pts_missing = points[missing_indices].cpu()
        pts_filled = points[filled_indices].cpu()
        
        if len(pts_filled) > 0:
            distances = torch.cdist(pts_missing, pts_filled, p=2)
            closest_vertex_indices = torch.argmin(distances, dim=1).cpu()
            ft_per_point[missing_indices, :] = ft_per_point[filled_indices][closest_vertex_indices, :]
        else:
            print("Warning: No features were extracted! Check render settings/radius.")
        
    t2 = time() - t1
    print(f"Time taken in mins: {t2 / 60:.2f}")
    return ft_per_point