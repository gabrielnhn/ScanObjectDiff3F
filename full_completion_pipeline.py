import torch
import numpy as np
import os
from time import time
from tqdm import tqdm
import cv2 as cv
from PIL import Image

RESOLUTION = 512

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
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from unittest.mock import MagicMock
sys.modules['nvdiffrast'] = MagicMock()
sys.modules['nvdiffrast.torch'] = MagicMock()

sys.path.append("./instantmesh")
import models.lrm_mesh
from utils.camera_util import get_zero123plus_input_cameras

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

    # Raster settings - image_size 512 for better contour precision
    image_size = RESOLUTION
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
    print(best_elev)
    print(best_azim)
    
    bbox = pcd.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bb_diff = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) / 2.0
    distance = torch.sqrt((bb_diff * bb_diff).sum()) * 0.65
    
    azimuths = [best_azim]
    elevations = [best_elev]
    
    R, T = look_at_view_transform(dist=distance, elev=torch.tensor(elevations, device=device), 
                                  azim=torch.tensor(azimuths, device=device), device=device, 
                                  at=bbox_center.unsqueeze(0))
    
    # cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=60.0)
    cameras = PerspectiveCameras(device=device, R=R, T=T)
    
    raster_settings = PointsRasterizationSettings(
        image_size=(H, W),
        radius=0.01,
        points_per_pixel=1,
        bin_size=0)
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    
    renderer = PhongCircleRenderer(background_color=(0.0,0.0,0.0)).to(device)
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
    
    # t1 = time()
    # best_elev = 12.016324043273926
    # best_azim = -129.30612182617188
    # exit()
    print("Rendering PyTorch3D Reference Image and Depth...")
    # Unpack both the images and the depth tensor
    batched_imgs, depth_tensor = render_with_pytorch3d(device, pcd, best_elev, best_azim)
    
    ref_rgb = batched_imgs[0].cpu().numpy()
    ref_rgb = (ref_rgb * 255).astype(np.uint8)
    import cv2
    # cv2.imwrite(os.path.join(renders_dir, "REFERENCE-rgb.png"), ref_rgb)
    # exit()
        
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

    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2", 
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )

    # Load the custom white-background UNet from InstantMesh authors
    unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh",
                                    filename="diffusion_pytorch_model.bin",
                                    repo_type="model",
                                    cache_dir=model_cache_dir)

    pipeline.unet.load_state_dict(torch.load(unet_ckpt_path, map_location='cpu'), strict=True)
    pipeline = pipeline.to(device)

    print("   Generating 6 multi-views...")
    z123_image = pipeline(processed_image, num_inference_steps=50).images[0]
    # z123_image.save(shape+"-zero123.png")
    #exit()
    print("   Flushing Zero123++ from VRAM...")
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return z123_image


def get_imesh_triplane(mv_image):
    
    # Convert the 960x640 grid directly into a [6, 3, 320, 320] tensor using einops (Zero cropping mistakes!)
    images_arr = np.asarray(mv_image, dtype=np.float32) / 255.0
    images_tensor = torch.from_numpy(images_arr).permute(2, 0, 1).contiguous()
    images_tensor = rearrange(images_tensor, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)

    # Batch it and cast to FP16
    image_tensor = images_tensor.unsqueeze(0).to(device, dtype=torch.float16)
    cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device, dtype=torch.float16)

    print("4. Loading InstantMesh...")
    model_ckpt_path = hf_hub_download(
        repo_id="TencentARC/InstantMesh",
        filename="instant_mesh_base.ckpt",
        repo_type="model",
        cache_dir=model_cache_dir
    )
    # grid_res=64 prevents the SDF OOM crash!
    model = models.lrm_mesh.InstantMesh(grid_res=64)
    state_dict = torch.load(model_ckpt_path, map_location='cpu', weights_only=True)['state_dict']
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.') and 'source_camera' not in k}
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device, dtype=torch.float16)
    model.init_flexicubes_geometry(device)

    print("   Extracting 3D Geometry...")
    with torch.no_grad():
        planes = model.forward_planes(image_tensor, cameras)
        
        # Flush memory right before the FlexiCubes SDF step
        torch.cuda.empty_cache()
        
        # (Note: make sure you kept your .float() casting fix inside flexicubes_geometry.py!)
        mesh_v, mesh_f, _, _, _, _ = model.get_geometry_prediction(planes)
        vertices = mesh_v[0]

    points = vertices.detach().cpu().numpy()

    def save_point_cloud_to_ply(points, filename):
        with open(filename, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for p in points:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

    output_filename = "IMESH.ply"
    save_point_cloud_to_ply(points, output_filename)
    print(f"Success! Point cloud saved to {output_filename}")

from pytorch3d.loss import chamfer_distance

def compute_metric(p1, p2):
    cd_l1_raw, _ = chamfer_distance(p1, p2, norm=1, point_reduction='mean')
    cd_l1 = cd_l1_raw * 100  # Scaling by 10^2

    return cd_l1.item()

def resample_pcd(pcd_np, n_points=16384):
    """
    Standardizes point cloud resolution to 16,384 points[cite: 1].
    """
    if len(pcd_np) > n_points:
        idx = np.random.choice(len(pcd_np), n_points, replace=False)
        return pcd_np[idx]
    elif len(pcd_np) < n_points:
        idx = np.random.choice(len(pcd_np), n_points, replace=True)
        return pcd_np[idx]
    return pcd_np


if __name__ == "__main__":
    print("----------")
    device = torch.device("cuda")
    dataset_path = "/home/gabrielnhn/datasets/synthetic_redwood/upload/plyobj"    
    # object = "horse.ply"
    object = "stanford-bunny.ply"
    
    renders_dir = os.path.join(renders_dir, object.split("."))
    # renders_dir = "renders"
    if not os.path.isdir(renders_dir):
        os.mkdir(renders_dir)
    
    from pc_utils import load_ply_to_pytorch3d 
    print("LOADING PCD;")
    partial_pcd = load_ply_to_pytorch3d(os.path.join(dataset_path, "indata", object),
                                        normal_factor=7)
    
    print("FIND AZIM/ELEV;")
    best_elev, best_azim = find_best_reference_pov_full(partial_pcd)
    print("GET BEST RGB;")
    canonical_image = get_reference_image(partial_pcd, best_elev, best_azim)
    print("RUN ZERO123++;")
    mv_image = get_mv_images(canonical_image)
    print("RUN INSTANTMESH FORWARD PASS;")
    out_points = get_imesh_triplane(mv_image)
    
    