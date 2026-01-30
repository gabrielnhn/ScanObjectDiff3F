import sys
sys.path.append("./Diff3F/")

import torch
import numpy as np
import trimesh

# from diff3f_pc import get_features_per_point
from diff3f_depthonly import get_features_per_point
from pc_utils import load_scanobjectnn_to_pytorch3d 
from utils import cosine_similarity, get_colors
from diffusion import init_pipe
from dino import init_dino

device = torch.device('cuda:0')
torch.cuda.set_device(device)

num_views = 20
num_views = 20
H = 512
W = 512
num_images_per_prompt = 1
tolerance = 0.004
use_normal_map = True

def compute_pc_features(device, pipe, dino_model, pcd, prompt):
    features = get_features_per_point(
        device=device,
        pipe=pipe, 
        dino_model=dino_model,
        pcd=pcd, 
        prompt=prompt,
        num_views=num_views,
        H=H,
        W=W,
        tolerance=tolerance,
        num_images_per_prompt=num_images_per_prompt,
        use_normal_map=use_normal_map,
    )
    return features.cpu()

def save_ply_with_colors(points, colors, filename):
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    
    pcd = trimesh.PointCloud(vertices=points, colors=colors)
    pcd.export(filename)
    print(f"Saved {filename}")

pipe = init_pipe(device)
dino_model = init_dino(device)

SOURCE_FILE = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/pillow/014_00015.bin"
TARGET_FILE = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/pillow/scene0271_00_00019.bin" 

print("Loading Point Clouds...")
source_pcd = load_scanobjectnn_to_pytorch3d(SOURCE_FILE, device)
target_pcd = load_scanobjectnn_to_pytorch3d(TARGET_FILE, device)

print("computing features for source (pillow a)...")
f_source = compute_pc_features(device, pipe, dino_model, source_pcd, "a pillow")

print("computing features for target (pillow b)...")
f_target = compute_pc_features(device, pipe, dino_model, target_pcd, "a pillow")

print("Calculating Correspondence...")
sim_matrix = cosine_similarity(f_source.to(device), f_target.to(device))

s = torch.argmax(sim_matrix, dim=0).cpu().numpy()

source_points_np = source_pcd.points_padded()[0].cpu().numpy()
target_points_np = target_pcd.points_padded()[0].cpu().numpy()

source_colors_np = source_pcd.features_padded()[0].cpu().numpy()

source_colors_np = get_colors(source_points_np) 

target_colors_mapped = source_colors_np[s]

save_ply_with_colors(source_points_np, source_colors_np, "results_source.ply")
save_ply_with_colors(target_points_np, target_colors_mapped, "results_target_correspondence.ply")

trimesh.load("results_source.ply").show()
trimesh.load("results_target_correspondence.ply").show()