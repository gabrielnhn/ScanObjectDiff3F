import torch
import numpy as np
import os
from time import time
from tqdm import tqdm

from pc_utils import load_scanobjectnn_to_pytorch3d, save_pointcloud_with_features
import zero123_diffusion

RESOLUTION = 320
DEFAULT_TEXT_PROMPT = ""
PROMPT_APPEND_ALWAYS = ", high quality, best quality"

from pytorch3d.renderer import (
    look_at_view_transform,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
    FoVPerspectiveCameras
)
from pytorch3d.ops import knn_points

import torchvision
TPL = torchvision.transforms.ToPILImage
tpl = TPL()

device = torch.device("cuda")

if not os.path.isdir("renders"):
    os.mkdir("renders")

import torch.nn as nn
import torch.nn.functional as F

class PhongCircleRenderer(nn.Module):
    """ Render circles with Blinn-Phong shading. """
    def __init__(self, background_color=(1.0, 1.0, 1.0), ambient=0.6, diffuse=0.8):
        super().__init__()
        self.compositor = AlphaCompositor(background_color=background_color)
        self.ambient = ambient
        self.diffuse = diffuse

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
        n_dot_l = torch.where(n_dot_l < 0, -n_dot_l, n_dot_l)
                
        diffuse_term = torch.clamp(n_dot_l, min=0.0)
        shaded_features = features * (self.ambient + self.diffuse * diffuse_term)
        shaded_features = torch.clamp(shaded_features, 0.0, 1.0)
        shaded_features = shaded_features.permute(1, 0)

        images = self.compositor(indices, weights, shaded_features)
        return images.permute(0, 2, 3, 1)


def find_best_reference_pov(pcd, device, pose_w=0.5):
    print("Searching for Canonical Front Face (ComPC Metric)...")
    points = pcd.points_padded()[0]
    total_points = points.shape[0]
    
    bbox = pcd.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bbox_center = (bbox_min + bbox_max) / 2.0
    distance = torch.sqrt(((bbox_max - bbox_min) ** 2).sum()) * 0.65

    raster_settings = PointsRasterizationSettings(image_size=(128, 128), radius=0.03, points_per_pixel=1)

    startv, endv = -80.0, 80.0
    starth, endh = -180.0, 180.0
    num = 40 
    batch_size = 16
    best_final_elev, best_final_azim = 0.0, 0.0

    for j in range(2):
        print(f"  -> Running Search Pass {j+1}/2...")
        vers = torch.linspace(startv, endv, num, device=device)
        hors = torch.linspace(starth, endh, num, device=device)
        verss, horss = torch.meshgrid(vers, hors, indexing='ij')
        verss, horss = verss.flatten(), horss.flatten()
        
        best_loss = float('inf')
        best_elev, best_azim = 0.0, 0.0
        
        for i in range(0, len(verss), batch_size):
            chunk_elevs = verss[i:i+batch_size]
            chunk_azims = horss[i:i+batch_size]
            
            R, T = look_at_view_transform(dist=distance, elev=chunk_elevs, azim=chunk_azims, device=device, at=bbox_center.unsqueeze(0))
            cameras = PerspectiveCameras(device=device, R=R, T=T)
            rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
            
            pcd_batch = pcd.extend(len(chunk_elevs))
            fragments = rasterizer(pcd_batch)
            idx_map = fragments.idx[..., 0] 
            cam_centers = cameras.get_camera_center() 
            
            for b in range(len(chunk_elevs)):
                valid_mask = idx_map[b] != -1
                visible_indices = torch.unique(idx_map[b][valid_mask])
                
                if len(visible_indices) > 0:
                    visible_indices = visible_indices % total_points
                    visible_points = points[visible_indices] 
                    dists, _, _ = knn_points(points.unsqueeze(0), visible_points.unsqueeze(0), K=1)
                    fixed_cd = dists.squeeze().sqrt().mean() 
                    cen = cam_centers[b]
                    posedist = (cen.unsqueeze(0) - points).square().sum(-1).sqrt().mean()
                    loss = fixed_cd + (pose_w * posedist)
                else:
                    loss = torch.tensor(float('inf'))
                
                if loss < best_loss:
                    best_loss = loss
                    best_elev = chunk_elevs[b].item()
                    best_azim = chunk_azims[b].item()
                    
            del pcd_batch, fragments, cameras, rasterizer
            torch.cuda.empty_cache() 

        interv = (endv - startv) / num
        interh = (endh - starth) / num
        startv, endv = best_elev - interv, best_elev + interv
        starth, endh = best_azim - interh, best_azim + interh
        best_final_elev, best_final_azim = best_elev, best_azim

    print(f"Optimal POV Found -> Azimuth: {best_final_azim:.1f}°, Elevation: {best_final_elev:.1f}°")
    return best_final_elev, best_final_azim

def render_with_pytorch3d(device, pcd, best_elev, best_azim, H=RESOLUTION, W=RESOLUTION):
    bbox = pcd.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bb_diff = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) / 2.0
    distance = torch.sqrt((bb_diff * bb_diff).sum()) * 1.4
    
    azimuths = [best_azim]
    elevations = [best_elev]
    
    R, T = look_at_view_transform(dist=distance, elev=torch.tensor(elevations, device=device), 
                                  azim=torch.tensor(azimuths, device=device), device=device, 
                                  at=bbox_center.unsqueeze(0))
    
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=30.0)
    
    raster_settings = PointsRasterizationSettings(image_size=(H, W), radius=0.02, points_per_pixel=1, bin_size=0)
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    
    renderer = PhongCircleRenderer(background_color=(0.5,0.5,0.5)).to(device)
    fragments = rasterizer(pcd)
    images = renderer(fragments, pcd).cpu()
    
    return images


def get_diffused_depth(pcd, path_append="", text_prompt=None):
    renders_dir = os.path.join("renders", f"usingZero123"+path_append)
    if not os.path.isdir(renders_dir):
        os.mkdir(renders_dir)    
    
    if text_prompt is None:
        text_prompt = DEFAULT_TEXT_PROMPT
    text_prompt += PROMPT_APPEND_ALWAYS
    
    t1 = time()

    print("Finding optimal reference viewpoint...")
    best_elev, best_azim = find_best_reference_pov(pcd, device)

    print("Rendering PyTorch3D Reference Image...")
    batched_imgs = render_with_pytorch3d(device, pcd, best_elev, best_azim)
    best_pov_image = batched_imgs[0].permute(2, 0, 1)
    best_pov_image = tpl(best_pov_image)
    best_pov_image.save(os.path.join(renders_dir, "REFERENCE.png"))

    # DIFFUSION STEP
    # base_pipe, normal_pipe = zero123_diffusion.init_pipelines()
    
    genimg, normalimg = zero123_diffusion.run_diffusion(
        # base_pipe,        
        # normal_pipe,
        best_pov_image,       
        text_prompt=text_prompt,
    )
    
    from matting_postprocess import postprocess
    print("Running matting postprocess...")
    genimg, normalimg = postprocess(genimg, normalimg)

    genimg.save(os.path.join(renders_dir, "COLORS_GRID.png"))
    normalimg.save(os.path.join(renders_dir, "NORMALS_GRID.png"))
    
    print(f"Time taken: {(time() - t1) / 60:.2f} min")
    
if __name__ == "__main__":
    print("----------")
    device = torch.device("cuda")
    dataset_path = "/home/gabrielnhn/datasets/synthetic_redwood/upload/plyobj"    
    object = "horse.ply"
    
    from pc_utils import load_ply_to_pytorch3d 
    partial_pcd = load_ply_to_pytorch3d(os.path.join(dataset_path, "indata", object))
    
    path_name = f"RedWood"
    get_diffused_depth(partial_pcd, path_append=path_name,
        text_prompt="horse, complete horse"
    )