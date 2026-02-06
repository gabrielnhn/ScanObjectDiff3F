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

first_points_np = first_pcd.points_padded()[0].cpu().numpy()
second_points_np = second_pcd.points_padded()[0].cpu().numpy()

if torch.is_tensor(first_labels):
    source_labels = first_labels.cpu().numpy().astype(int)
else:
    source_labels = first_labels.astype(int)

target_labels_pred = source_labels[s]

palette = np.array([
    [255, 0, 0],    
    [0, 255, 0],    
    [0, 0, 255],    
    [255, 255, 0],  
    [0, 255, 255],  
    [200, 200, 200] 
])

def labels_to_rgb(labels, palette):
    safe_labels = labels.copy()
    safe_labels[safe_labels >= len(palette)-1] = len(palette) - 1 
    return palette[safe_labels]

print("Saving results...")

# Colorize Source (Ground Truth)
c_source = labels_to_rgb(source_labels, palette)
save_ply_with_colors(first_points_np, c_source, "results_source_gt.ply")

# Colorize Target (Prediction via One-Shot Transfer)
c_target = labels_to_rgb(target_labels_pred, palette)
save_ply_with_colors(second_points_np, c_target, "results_target_transfer.ply")

print("Done! Open 'results_source_gt.ply' and 'results_target_transfer.ply'.")