import sys
sys.path.append("./Diff3F/")

import torch
import numpy as np
import trimesh

# --- Import your new modules ---
from diff3f_pc import get_features_per_point
from pc_utils import load_scanobjectnn_to_pytorch3d 
from utils import cosine_similarity, get_colors
from diffusion import init_pipe
from dino import init_dino

# --- Configuration ---
device = torch.device('cuda:0')
torch.cuda.set_device(device)

num_views = 20 # Reduced for testing, increase to 40-100 for best results
H = 512
W = 512
num_images_per_prompt = 1
tolerance = 0.004
use_normal_map = True

# --- 1. Wrapper to compute features for Point Clouds ---
def compute_pc_features(device, pipe, dino_model, pcd, prompt):
    # We pass the PyTorch3D Pointclouds object (pcd) directly
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

# --- 2. Helper to save PLY files for visualization ---
def save_ply_with_colors(points, colors, filename):
    """
    points: (N, 3) numpy array
    colors: (N, 3) numpy array (0-1 or 0-255)
    """
    # Ensure colors are 0-255 uint8
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    
    # Use trimesh to save easily
    pcd = trimesh.PointCloud(vertices=points, colors=colors)
    pcd.export(filename)
    print(f"Saved {filename}")

# --- Main Execution ---

# Initialize Models
pipe = init_pipe(device)
dino_model = init_dino(device)

# File Paths (Change these to your actual files)
# Example: Using two different pillows to see if corners match corners
SOURCE_FILE = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/pillow/014_00015.bin"
TARGET_FILE = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/pillow/scene0271_00_00019.bin" 

print("Loading Point Clouds...")
# Load into PyTorch3D objects (includes RGB/Normals if available)
source_pcd = load_scanobjectnn_to_pytorch3d(SOURCE_FILE, device)
target_pcd = load_scanobjectnn_to_pytorch3d(TARGET_FILE, device)

print("Computing Features for Source (Pillow A)...")
f_source = compute_pc_features(device, pipe, dino_model, source_pcd, "a pillow")

print("Computing Features for Target (Pillow B)...")
f_target = compute_pc_features(device, pipe, dino_model, target_pcd, "a pillow")

# --- 3. Correspondence Logic ---
print("Calculating Correspondence...")
# Compute Cosine Similarity between all points
# Shapes: f_source (N_src, Dim), f_target (N_tgt, Dim)
sim_matrix = cosine_similarity(f_source.to(device), f_target.to(device))

# Find the best match in Source for every point in Target
# s contains indices: "For target point i, which source point j is most similar?"
s = torch.argmax(sim_matrix, dim=0).cpu().numpy()

# --- 4. Color Transfer (Visual Proof) ---
# Extract raw points for saving
source_points_np = source_pcd.points_padded()[0].cpu().numpy()
target_points_np = target_pcd.points_padded()[0].cpu().numpy()

# Get Source Colors (Gradient or Real Colors)
# Option A: Use the real colors from the scan
source_colors_np = source_pcd.features_padded()[0].cpu().numpy()

# Option B: Generate a rainbow gradient (Spectral) to make correspondence obvious
# This is better for checking if "left matches left" and "right matches right"
print("Generating Spectral Map for visualization...")
source_colors_np = get_colors(source_points_np) 

# Transfer colors: Target gets the color of its matching Source point
target_colors_mapped = source_colors_np[s]

# --- 5. Save Results ---
save_ply_with_colors(source_points_np, source_colors_np, "results_source.ply")
save_ply_with_colors(target_points_np, target_colors_mapped, "results_target_correspondence.ply")

print("Done! Open 'results_source.ply' and 'results_target_correspondence.ply' in MeshLab or CloudCompare.")
print("If the correspondence worked, the Pillow B should look colored exactly like Pillow A.")

# Optional: View immediately if running locally
try:
    trimesh.load("results_source.ply").show()
    trimesh.load("results_target_correspondence.ply").show()
except:
    pass