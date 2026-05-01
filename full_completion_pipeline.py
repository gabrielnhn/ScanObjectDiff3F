import torch
import numpy as np
import os
from time import time
from tqdm import tqdm
import cv2 as cv
from PIL import Image

RESOLUTION = 512
# CANON_ONLY = False
CANON_ONLY = True
# RESOLUTION = 320

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

import os
import sys
import gc
import torch
import numpy as np
from PIL import Image
import rembg
from einops import rearrange
from huggingface_hub import hf_hub_download
from diffusers import (
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler
)
from unittest.mock import MagicMock
sys.modules['nvdiffrast'] = MagicMock()
sys.modules['nvdiffrast.torch'] = MagicMock()

sys.path.append("./instantmesh")
import models.lrm_mesh
from utils.camera_util import (
    get_zero123plus_input_cameras,
    FOV_to_intrinsics,
    spherical_camera_pose
)

device = "cuda"
model_cache_dir = './ckpts/'
os.makedirs(model_cache_dir, exist_ok=True)

# renders_dir = os.path.join("renders", path_append)
renders_dir = "renders"
if not os.path.isdir(renders_dir):
    os.mkdir(renders_dir)    

def find_best_reference_pov_full(pcd, pose_w=0.5, edge_w=0.1):
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
    # distance = 4.0

    image_size = 512
    raster_settings = PointsRasterizationSettings(
        image_size=image_size, 
        radius=0.01, 
        points_per_pixel=1,
        bin_size=0
    )

    startv, endv = -80.0, 80.0
    starth, endh = -180.0, 180.0
    num = 50 
    # num = 20 
    # num = 40 
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
            # cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=30.0)
            
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
                    visible_indices = visible_indices % total_points
                    visible_pts = points[visible_indices] 
                    
                    # Chamfer-like fidelity
                    dists, _, _ = knn_points(points.unsqueeze(0), visible_pts.unsqueeze(0), K=1)
                    fixed_cd = dists.squeeze().sqrt().mean() 
                    
                    # Pose distance regularization
                    posedist = (cam_centers[b].unsqueeze(0) - points).square().sum(-1).sqrt().mean()
                    
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
                    # import random
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
        #     cv.imwrite(os.path.join(
        #                 renders_dir, f"combined-{j}-{best_elev:.1f}-{best_azim:.1f}.jpg")
        #                ,
        #                best_img_to_save)

    print(f"Optimal POV Found -> Azimuth: {best_final_azim:.1f}°, Elevation: {best_final_elev:.1f}°")
    return best_final_elev, best_final_azim

def render_with_pytorch3d(device, pcd, best_elev, best_azim, H=RESOLUTION, W=RESOLUTION):
    print(best_elev)
    print(best_azim)
    
    bbox = pcd.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bb_diff = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) / 2.0
    # distance = 4.0
    distance = torch.sqrt((bb_diff * bb_diff).sum()) * 0.65
    
    azimuths = [best_azim]
    elevations = [best_elev]
    
    R, T = look_at_view_transform(dist=distance, elev=torch.tensor(elevations, device=device), 
                                  azim=torch.tensor(azimuths, device=device), device=device, 
                                  at=bbox_center.unsqueeze(0))
    
    # cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=30.0)
    cameras = PerspectiveCameras(device=device, R=R, T=T)
    
    # raster_settings = PointsRasterizationSettings(
    #     image_size=(H, W),
    #     radius=0.01,
    #     points_per_pixel=1,
    #     bin_size=0)
    # 3. FIXED RESOLUTION (320x320 from ValidationData)
    raster_settings = PointsRasterizationSettings(
        image_size=(H, W),
        radius=0.01,
        points_per_pixel=1,
        bin_size=0
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    
    renderer = PhongCircleRenderer(background_color=(0.0,0.0,0.0)).to(device)
    # renderer = PhongCircleRenderer(background_color=(1.0,1.0,1.0)).to(device)
    # renderer = NormalsRenderer(
    #     # background_color=(0.5,0.5,0.5),
    #     background_color=(0.0,0.0,0.0),
    #     cameras=cameras).to(device)
    
    fragments = rasterizer(pcd)
    images = renderer(fragments, pcd).cpu()
    
    depth = fragments.zbuf[..., 0].cpu()
    valid_mask = (fragments.idx[..., 0] != -1).cpu()
    depth[~valid_mask] = -1
    
    return images, depth

def get_reference_image(pcd, best_elev, best_azim):
    print("Rendering PyTorch3D Reference Image and Depth...")
    # Unpack both the images and the depth tensor
    batched_imgs, depth_tensor = render_with_pytorch3d(device, pcd, best_elev, best_azim)
    
    ref_rgb = batched_imgs[0].cpu().numpy()
    ref_rgb = (ref_rgb * 255).astype(np.uint8)
    # import cv
    ref_alpha = torch.zeros_like(depth_tensor[0], dtype=torch.uint8)
    ref_alpha[depth_tensor[0] > 0] = 255
    ref_alpha = ref_alpha.cpu().numpy()[..., None] # Add channel dimension
    ref_rgba = np.concatenate([ref_rgb, ref_alpha], axis=-1)
    
    best_pov_image = Image.fromarray(ref_rgba, mode='RGBA')
    best_pov_image.save(os.path.join(renders_dir, "REFERENCE-post.png"))
    
    return best_pov_image
    
    
def get_mv_images(canonical_img):
    # Remove background
    # raw_img = Image.open(input_image_path)
    no_bg_img = rembg.remove(canonical_img)

    # Paste onto a PURE WHITE background (CRITICAL FOR INSTANTMESH)
    white_bg = Image.new("RGBA", no_bg_img.size, "WHITE")
    white_bg.paste(no_bg_img, (0, 0), mask=no_bg_img)
    processed_image = white_bg.convert("RGB")
    processed_image = processed_image.resize((320, 320)) # Ensure standard size

    if CANON_ONLY:
        return None, processed_image

    generator = torch.Generator(device=device).manual_seed(42)

    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2", 
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16,
    )
    # pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler = EulerDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )

    # Load the custom white-background UNet from InstantMesh authors
    unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh",
                                    filename="diffusion_pytorch_model.bin",
                                    repo_type="model",
                                    cache_dir=model_cache_dir)

    pipeline.unet.load_state_dict(torch.load(unet_ckpt_path, map_location='cpu'), strict=True)
    pipeline = pipeline.to(device)

    print("Zero123++...")
    z123_image = pipeline(
        processed_image,
        num_inference_steps=50, 
        guidance_scale=7.5,
        generator=generator,
        # prompt="high quality, clay material, complete",
        # negative_prompt="complex, detailed, chaotic, asymmetric, text, logo",
    ).images[0]
    z123_image.save(os.path.join(renders_dir, "MV.png"))
    
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return z123_image, processed_image


def get_imesh_triplane(
    mv_image,
    processed_canonical_image,
    best_elev,
    best_azim,
):
    
    if not CANON_ONLY:
        # Convert the 960x640 grid directly into a [6, 3, 320, 320] tensor using einops (Zero cropping mistakes!)
        images_arr = np.asarray(mv_image, dtype=np.float32) / 255.0
        images_tensor = torch.from_numpy(images_arr).permute(2, 0, 1).contiguous()
        images_tensor = rearrange(images_tensor, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)
        # Batch it and cast to FP16
        image_tensor = images_tensor.unsqueeze(0).to(device, dtype=torch.float16)
        cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device, dtype=torch.float16)

    canon_arr = np.asarray(processed_canonical_image, dtype=np.float32) / 255.0
    canon_tensor = torch.from_numpy(canon_arr).permute(2, 0, 1).contiguous()
    canon_tensor = canon_tensor.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float16) # [1, 1, 3, 320, 320]

    canon_c2w = spherical_camera_pose(np.array([0.0]), np.array([0.0]), radius=4.0)
    canon_c2w = canon_c2w.float().flatten(-2) # [1, 16]
    canon_K = FOV_to_intrinsics(60.0).unsqueeze(0).float().flatten(-2) # [1, 9]

    # Combine Extrinsics and Intrinsics exactly like InstantMesh does
    canon_ext = canon_c2w[:, :12]
    canon_int = torch.stack([canon_K[:, 0], canon_K[:, 4], canon_K[:, 2], canon_K[:, 5]], dim=-1)
    canon_cam = torch.cat([canon_ext, canon_int], dim=-1)
    canon_cam = canon_cam.unsqueeze(0).to(device, dtype=torch.float16) # [1, 1, 16]

    # # Append to the Zero123++ cameras
    if CANON_ONLY:
        image_tensor = canon_tensor
        cameras = canon_cam
    else:
        image_tensor = torch.cat([image_tensor, canon_tensor], dim=1) # Now [1, 7, 3, 320, 320]
        cameras = torch.cat([cameras, canon_cam], dim=1) # Now [1, 7, 16]

    print("Loading InstantMesh...")
    model_ckpt_path = hf_hub_download(
        repo_id="TencentARC/InstantMesh",
        filename="instant_mesh_base.ckpt",
        repo_type="model",
        cache_dir=model_cache_dir
    )
    
    model = models.lrm_mesh.InstantMesh(grid_res=64)
    state_dict = torch.load(model_ckpt_path, map_location='cpu', weights_only=True)['state_dict']
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.') and 'source_camera' not in k}
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device, dtype=torch.float16)
    model.init_flexicubes_geometry(device)

    with torch.inference_mode():
        planes = model.forward_planes(image_tensor, cameras)
        torch.cuda.empty_cache()
        mesh_v, mesh_f, _, _, _, _ = model.get_geometry_prediction(planes)
        vertices = mesh_v[0]

    points = vertices.detach().cpu().numpy()
    return points


from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Pointclouds

def compute_metric(p1, p2):
    p1 = Pointclouds(torch.tensor(p1).float().unsqueeze(0))
    p2 = Pointclouds(torch.tensor(p2).float().unsqueeze(0))
    cd_l1_raw, _ = chamfer_distance(p1, p2, norm=1, point_reduction='mean')
    cd_l1 = cd_l1_raw * 100  # Scaling by 10^2
    return cd_l1.item()

def resample_pcd(pcd_np, n_points=16384):
    if len(pcd_np) > n_points:
        idx = np.random.choice(len(pcd_np), n_points, replace=False)
        return pcd_np[idx]
    elif len(pcd_np) < n_points:
        idx = np.random.choice(len(pcd_np), n_points, replace=True)
        return pcd_np[idx]
    return pcd_np


import open3d as o3d
import copy

def normalize_pc(pcd_np):
    """Centers the point cloud at the origin and scales it to a unit bounding box."""
    centroid = np.mean(pcd_np, axis=0)
    pcd_np_centered = pcd_np - centroid
    max_distance = np.max(np.sqrt(np.sum(pcd_np_centered**2, axis=1)))
    if max_distance == 0: 
        max_distance = 1.0
    pcd_np_normalized = pcd_np_centered / max_distance
    return pcd_np_normalized

def run_icp(source_np, target_np):
    """Runs Open3D ICP to perfectly align source to target."""
    source_o3d = o3d.geometry.PointCloud()
    source_o3d.points = o3d.utility.Vector3dVector(source_np)
    
    target_o3d = o3d.geometry.PointCloud()
    target_o3d.points = o3d.utility.Vector3dVector(target_np)

    # Point-to-point ICP
    threshold = 0.1 # Search radius (slightly larger to catch rotation offsets)
    trans_init = np.eye(4)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_o3d, target_o3d, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    
    source_o3d.transform(reg_p2p.transformation)
    return np.asarray(source_o3d.points)

def save_debug_ply(np_points, filename):
    import open3d as o3d
    debug_pcd = o3d.geometry.PointCloud()
    debug_pcd.points = o3d.utility.Vector3dVector(np_points)
    o3d.io.write_point_cloud(filename, debug_pcd)

if __name__ == "__main__":
    print("----------")
    device = torch.device("cuda")
    dataset_path = "/home/gabrielnhn/datasets/synthetic_redwood/upload/plyobj"    
    # object = "horse.ply"
    # object = "stanford-bunny.ply"
    object = "cow.ply"
    
    renders_dir = os.path.join(renders_dir, object.split(".")[0])
    if not os.path.isdir(renders_dir):
        os.mkdir(renders_dir)
    
    from pc_utils import load_ply_to_pytorch3d 
    print("LOADING PCD;")
    partial_pcd = load_ply_to_pytorch3d(os.path.join(dataset_path, "indata", object),
                                        normal_factor=5)
    
    angles = {
        "stanford-bunny.ply": (12.016324043273926, -129.30612182617188),
        "cow.ply": (10.44897747039795,-2.3510241508483887),
    }
    
    # print("FIND AZIM/ELEV;")
    if object in angles:
        best_elev, best_azim = angles[object]
    else:
        best_elev, best_azim = find_best_reference_pov_full(partial_pcd)
    
    # best_elev = 12.016324043273926
    # best_azim = -129.30612182617188

    print("GET BEST RGB;")
    canonical_image = get_reference_image(partial_pcd, best_elev, best_azim)
    
    print("RUN ZERO123++;")
    mv_image, resized_canonical = get_mv_images(canonical_image)
    print("RUN INSTANTMESH FORWARD PASS;")
    out_points = get_imesh_triplane(mv_image, resized_canonical, best_elev, best_azim)

    print("done. aligning..")    
    gt_pcd = load_ply_to_pytorch3d(os.path.join(dataset_path, "gtdata", object), normal_factor=0)

    gt_np = gt_pcd.points_list()[0].cpu().numpy()
    partial_np = partial_pcd.points_list()[0].cpu().numpy()
    out_np = out_points 

    gt_centroid = np.mean(gt_np, axis=0)
    gt_np_centered = gt_np - gt_centroid
    gt_scale = np.max(np.sqrt(np.sum(gt_np_centered**2, axis=1)))
    if gt_scale == 0: gt_scale = 1.0
    
    gt_np_norm = gt_np_centered / gt_scale
    partial_np_norm = (partial_np - gt_centroid) / gt_scale 
    out_np_norm = normalize_pc(out_np)
    
    mirror_matrix = np.array([
        [-1,  0,  0], 
        [ 0,  1,  0],
        [ 0,  0,  1]
    ])
    # Z-Up to Y-Up
    flip_matrix = np.array([
        [ 1,  0,  0],
        [ 0,  0, -1],
        [ 0,  1,  0]
    ])
    yaw_degrees = 90
    yaw_rad = np.radians(yaw_degrees)
    yaw_matrix = np.array([
        [ np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [ 0,               1, 0              ],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])
    M_offset = mirror_matrix @ flip_matrix @ yaw_matrix
    
    out_np_view_space = out_np_norm @ M_offset

    R, _ = look_at_view_transform(dist=1.0, elev=best_elev, azim=best_azim, device=device)
    R_np = R[0].cpu().numpy()
    
    # tranp=inverse
    out_np_deterministic = out_np_view_space @ R_np.T

    print("RESAMPLING FOR METRICS;")
    out_np_resampled = resample_pcd(out_np_deterministic, n_points=16384)
    partial_np_resampled = resample_pcd(partial_np_norm, n_points=16384)
    gt_np_resampled = resample_pcd(gt_np_norm, n_points=16384)

    print("SAVING DEBUG POINT CLOUDS;")
    
    if CANON_ONLY:
        save_debug_ply(out_np_resampled, os.path.join(renders_dir, "1view.ply"))
    else:
        save_debug_ply(out_np_resampled, os.path.join(renders_dir, "7views.ply"))
        
    save_debug_ply(gt_np_resampled, os.path.join(renders_dir, "debug_2_ground_truth.ply"))
    save_debug_ply(partial_np_resampled, os.path.join(renders_dir, "debug_3_partial_sensor.ply"))
    
    cd_score = compute_metric(out_np_resampled, gt_np_resampled)
    print(f"Final Chamfer Distance (Scaled): {cd_score:.4f}")