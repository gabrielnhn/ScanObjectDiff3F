import sys
sys.path.append("./Diff3F/")

import torch
import numpy as np
import trimesh

# from diff3f_pc import get_features_per_point
from diff3f_depthonly import get_features_per_point
from pc_utils import load_scanobjectnn_to_pytorch3d, save_pointcloud_with_features
from utils import cosine_similarity, get_colors
from diffusion import init_pipe
from dino import init_dino




device = torch.device('cuda:0')
torch.cuda.set_device(device)

# num_views = 20
# num_views = 100
num_views = 1
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
    )# .cpu()
    return features

def save_ply_with_colors(points, colors, filename):
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    
    pcd = trimesh.PointCloud(vertices=points, colors=colors)
    pcd.export(filename)
    print(f"Saved {filename}")

pipe = init_pipe(device)
dino_model = init_dino(device)

first_FILE = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/pillow/014_00015.bin"
second_FILE = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/pillow/scene0271_00_00019.bin" 

print("Loading Point Clouds...")
first_pcd, first_labels = load_scanobjectnn_to_pytorch3d(first_FILE, device)
second_pcd, second_labels = load_scanobjectnn_to_pytorch3d(second_FILE, device)

print("computing features for first (pillow a)...")
f_first = compute_pc_features(device, pipe, dino_model, first_pcd, "a pillow")

save_pointcloud_with_features(first_pcd, f_first, "pointcloud1_with_features")

print("computing features for second (pillow b)...")
f_second = compute_pc_features(device, pipe, dino_model, second_pcd, "a pillow")


save_pointcloud_with_features(second_pcd, f_second, "pointcloud2_with_features")


print("Calculating Correspondence...")
sim_matrix = cosine_similarity(f_first.to(device), f_second.to(device))

s = torch.argmax(sim_matrix, dim=0).cpu().numpy()

first_points_np = first_pcd.points_padded()#[0].cpu().numpy()
second_points_np = second_pcd.points_padded()#[0].cpu().numpy()

# first_colors_np = first_pcd.features_padded()[0].cpu().numpy()
# first_colors_np = get_colors(first_points_np) 
# second_colors_mapped = first_colors_np[s]
# trimesh.load("results_first.ply").show()
# trimesh.load("results_second_correspondence.ply").show()

palette = np.array([
        [255, 0, 0], 
        [0, 255, 0], 
        [0, 0, 255], 
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
    ])

save_ply_with_colors(first_points_np, palette[first_labels], "results_first.ply")
save_ply_with_colors(second_points_np, palette[first_labels[s]], "results_second_correspondence.ply")
