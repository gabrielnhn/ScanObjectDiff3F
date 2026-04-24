import torch
import numpy as np
import os
from time import time
from tqdm import tqdm
import cv2 as cv
from PIL import Image



from pc_utils import load_scanobjectnn_to_pytorch3d, save_pointcloud_with_features
import zero123_diffusion

RESOLUTION = 320
DEFAULT_TEXT_PROMPT = ""
PROMPT_APPEND_ALWAYS = ", realistic, high quality, best quality"

from pytorch3d.renderer import (
    look_at_view_transform,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
    FoVPerspectiveCameras
)
from pytorch3d.ops import knn_points


from renderers import PhongCircleRenderer, NormalsRenderer

import torchvision
TPL = torchvision.transforms.ToPILImage
tpl = TPL()

device = torch.device("cuda")

if not os.path.isdir("renders"):
    os.mkdir("renders")


def find_best_reference_pov_full(pcd, device, pose_w=0.5, edge_w=0.01):
    """
    Combines COMPC (Chamfer Distance + Depth Regularization) 
    with OpenCV Depth-Edge Contour detection.
    """
    points = pcd.points_padded()[0]
    total_points = points.shape[0]
    
    # Calculate scene bounds for camera placement
    bbox = pcd.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bbox_center = (bbox_min + bbox_max) / 2.0
    distance = torch.sqrt(((bbox_max - bbox_min) ** 2).sum()) * 0.65

    # Raster settings - image_size 512 for better contour precision
    image_size = 512
    raster_settings = PointsRasterizationSettings(
        image_size=image_size, 
        radius=0.02, 
        points_per_pixel=1
    )

    startv, endv = -80.0, 80.0
    starth, endh = -180.0, 180.0
    num = 40 
    batch_size = 16
    best_final_elev, best_final_azim = 0.0, 0.0

    for j in range(2):
        vers = torch.linspace(startv, endv, num, device=device)
        hors = torch.linspace(starth, endh, num, device=device)
        verss, horss = torch.meshgrid(vers, hors, indexing='ij')
        verss, horss = verss.flatten(), horss.flatten()
        
        best_loss = float('inf')
        best_elev, best_azim = 0.0, 0.0
        best_img_to_save = None
        
        for i in range(0, len(verss), batch_size):
            chunk_elevs = verss[i:i+batch_size]
            chunk_azims = horss[i:i+batch_size]
            
            R, T = look_at_view_transform(dist=distance, elev=chunk_elevs, azim=chunk_azims, device=device, at=bbox_center.unsqueeze(0))
            cameras = PerspectiveCameras(device=device, R=R, T=T)
            rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
            
            pcd_batch = pcd.extend(len(chunk_elevs))
            fragments = rasterizer(pcd_batch)
            
            depth_maps = fragments.zbuf[..., 0] 
            idx_map = fragments.idx[..., 0] 
            cam_centers = cameras.get_camera_center() 
            
            for b in range(len(chunk_elevs)):
                valid_mask = idx_map[b] != -1
                visible_indices = torch.unique(idx_map[b][valid_mask])
                
                if len(visible_indices) > 0:
                    # --- 1. COMPC GEOMETRIC LOSS ---
                    visible_indices = visible_indices % total_points
                    visible_pts = points[visible_indices] 
                    
                    # Chamfer-like fidelity
                    dists, _, _ = knn_points(points.unsqueeze(0), visible_pts.unsqueeze(0), K=1)
                    fixed_cd = dists.squeeze().sqrt().mean() 
                    
                    # Pose distance regularization
                    posedist = (cam_centers[b].unsqueeze(0) - points).square().sum(-1).sqrt().mean()
                    
                    # --- 2. DEPTH-EDGE CONTOUR LOSS ---
                    d_map = depth_maps[b].clone()
                    max_val = d_map[valid_mask].max() if valid_mask.any() else 1.0
                    d_map[~valid_mask] = max_val * 1.2
                    
                    # Normalize for CV
                    d_min, d_max = d_map.min(), d_map.max()
                    d_norm = (d_map - d_min) / (d_max - d_min + 1e-6)
                    d_img = (d_norm.cpu().numpy() * 255).astype(np.uint8)
                    
                    edges = cv.Canny(d_img, 50, 150)
                    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
                    edge_score = len(contours)
                    
                    # --- 3. TOTAL WEIGHTED LOSS ---
                    # fixed_cd: completeness, posedist: closeness, edge_score: topology
                    loss = fixed_cd + (pose_w * posedist) + (edge_w * edge_score)
                    # print(f"FIXCD {fixed_cd} posedist {posedist} edgescore {edge_score}")
                else:
                    loss = torch.tensor(float('inf'))
                
                if loss < best_loss:
                    best_loss = loss
                    best_elev = chunk_elevs[b].item()
                    best_azim = chunk_azims[b].item()
                    
                    # Prepare debug image (convert to BGR for colorful contours)
                    # color_img = cv.cvtColor(d_img, cv.COLOR_GRAY2BGR)
                    # for cnt in contours:
                    #     cv.drawContours(color_img, [cnt], -1, (random.randint(0,255), 
                    #                                           random.randint(0,255), 
                    #                                           random.randint(0,255)), 1)
                    # best_img_to_save = color_img
            
            del pcd_batch, fragments, cameras, rasterizer
            torch.cuda.empty_cache() 

        # Hierarchical search update
        interv = (endv - startv) / num
        interh = (endh - starth) / num
        startv, endv = best_elev - interv, best_elev + interv
        starth, endh = best_azim - interh, best_azim + interh
        best_final_elev, best_final_azim = best_elev, best_azim
        
        # if best_img_to_save is not None:
        #     cv.imwrite(f"reference_test/combined-{j}-{best_elev:.1f}-{best_azim:.1f}.jpg", best_img_to_save)

    print(f"Optimal POV Found -> Azimuth: {best_final_azim:.1f}°, Elevation: {best_final_elev:.1f}°")
    return best_final_elev, best_final_azim

def render_with_pytorch3d(device, pcd, best_elev, best_azim, H=RESOLUTION, W=RESOLUTION):
    bbox = pcd.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bb_diff = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) / 2.0
    # distance = torch.sqrt((bb_diff * bb_diff).sum()) * 1.4
    distance = torch.sqrt((bb_diff * bb_diff).sum()) * 0.75
    
    azimuths = [best_azim]
    elevations = [best_elev]
    
    R, T = look_at_view_transform(dist=distance, elev=torch.tensor(elevations, device=device), 
                                  azim=torch.tensor(azimuths, device=device), device=device, 
                                  at=bbox_center.unsqueeze(0))
    
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=60.0)
    
    raster_settings = PointsRasterizationSettings(image_size=(H, W), radius=0.02, points_per_pixel=1, bin_size=0)
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    
    renderer = PhongCircleRenderer(background_color=(0.5,0.5,0.5)).to(device)
    # renderer = NormalsRenderer(background_color=(0.5,0.5,0.5), cameras=cameras).to(device)
    
    fragments = rasterizer(pcd)
    images = renderer(fragments, pcd).cpu()
    
    depth = fragments.zbuf[..., 0].cpu()
    valid_mask = (fragments.idx[..., 0] != -1).cpu()
    depth[~valid_mask] = -1
    
    return images, depth


def get_diffused_depth(pcd, path_append="", text_prompt=None):
    renders_dir = os.path.join("renders", f"usingZero123"+path_append)
    if not os.path.isdir(renders_dir):
        os.mkdir(renders_dir)    
    
    if text_prompt is None:
        text_prompt = DEFAULT_TEXT_PROMPT
    text_prompt += PROMPT_APPEND_ALWAYS
    
    t1 = time()

    print("Finding optimal reference viewpoint...")
    best_elev, best_azim = find_best_reference_pov_full(pcd, device)

    # exit()
    # 
    # 
    # 

    print("Rendering PyTorch3D Reference Image and Depth...")
    # Unpack both the images and the depth tensor
    batched_imgs, depth_tensor = render_with_pytorch3d(device, pcd, best_elev, best_azim)
    
    # ref_rgb = batched_imgs[0].cpu().numpy()
    # ref_rgb = (ref_rgb * 255).astype(np.uint8)
    # import cv2
    # cv2.imwrite(os.path.join(renders_dir, "REFERENCE-rgb.png"), ref_rgb)
        
    # ref_alpha = torch.zeros_like(depth_tensor[0], dtype=torch.uint8)
    # ref_alpha[depth_tensor[0] > 0] = 255
    # ref_alpha = ref_alpha.cpu().numpy()[..., None] # Add channel dimension
    
    # ref_rgba = np.concatenate([ref_rgb, ref_alpha], axis=-1)
    
    
    # best_pov_image = Image.fromarray(ref_rgba, mode='RGBA')
    # best_pov_image.save(os.path.join(renders_dir, "REFERENCE-post.png"))
    # completed_prior_image = best_pov_image

    # exit()
    # completed_prior_image.save(os.path.join(renders_dir, "REFERENCE_PRIOR.png"))
    # completed_prior_image = Image.open("./manual-bunny.png")
    # completed_prior_image = Image.open("./totebag.png")
    
    # DIFFUSION STEP
    
    
    from common_controlnet import run_diffusion
    depth_2d = depth_tensor[0].clone().squeeze() # Shape: (H, W)
    ref_alpha = torch.zeros_like(depth_2d, dtype=torch.uint8)
    ref_alpha[depth_2d > 0] = 255
    ref_alpha_np = ref_alpha.cpu().numpy()[..., None] # Shape: (H, W, 1)
    
    print("Formatting Depth Map for ControlNet...")
    valid_mask_2d = depth_2d > 0
    
    if valid_mask_2d.sum() > 0:
        min_depth = depth_2d[valid_mask_2d].min()
        max_depth = depth_2d[valid_mask_2d].max()
        
        if max_depth > min_depth:
            depth_norm = (depth_2d - min_depth) / (max_depth - min_depth)
            depth_norm = 1.0 - depth_norm
        else:
            depth_norm = torch.ones_like(depth_2d)
            
        depth_norm[~valid_mask_2d] = 0.0
    else:
        print("[WARN] No valid pixels in depth map!")
        depth_norm = torch.zeros_like(depth_2d)

    depth_uint8 = (depth_norm * 255).cpu().numpy().astype(np.uint8)
    depth_rgb = np.stack([depth_uint8] * 3, axis=-1)
    depth_pil = Image.fromarray(depth_rgb)
    
    depth_pil.save(os.path.join(renders_dir, "REFERENCE-depth.png"))

    print("[INFO] Running ControlNet Depth to hallucinate RGB...")
    hallucinated_rgb = run_diffusion(
        text_prompt=text_prompt, 
        depth_image=depth_pil, 
        conditioning_scale=1.0
    )
    
    # Save the raw ControlNet output
    hallucinated_rgb.save(os.path.join(renders_dir, "REFERENCE-controlnet-raw.png"))
    hallucinated_np = np.array(hallucinated_rgb)
    final_rgba = np.concatenate([hallucinated_np, ref_alpha_np], axis=-1)
    completed_prior_image = Image.fromarray(final_rgba, mode='RGBA')
    completed_prior_image.save(os.path.join(renders_dir, "REFERENCE-post.png"))
    # print(f"[INFO] Success! Completed in {time() - t1:.2f} seconds.")
    
    
    
    
    # genimg, normalimg = zero123_diffusion.run_diffusion(
    #     # base_pipe,        
    #     # normal_pipe,
    #     completed_prior_image,       
    #     # text_prompt=text_prompt,
    # )

    # genimg.save(os.path.join(renders_dir, "COLORS_GRID.png"))
    # normalimg.save(os.path.join(renders_dir, "NORMALS_GRID.png"))
    
    # from matting_postprocess import postprocess
    # print("Running matting postprocess...")
    # genimg, normalimg = postprocess(genimg, normalimg)

    # genimg.save(os.path.join(renders_dir, "COLORS_GRID_post.png"))
    # normalimg.save(os.path.join(renders_dir, "NORMALS_GRID_post.png"))
    
    # print(f"Time taken: {(time() - t1) / 60:.2f} min")
    
if __name__ == "__main__":
    print("----------")
    device = torch.device("cuda")
    dataset_path = "/home/gabrielnhn/datasets/synthetic_redwood/upload/plyobj"    
    # object = "horse.ply"
    object = "stanford-bunny.ply"
    
    
    from pc_utils import load_ply_to_pytorch3d 
    partial_pcd = load_ply_to_pytorch3d(os.path.join(dataset_path, "indata", object))
    
    path_name = f"RedWood"
    get_diffused_depth(partial_pcd, path_append=path_name,
        text_prompt=""
    )