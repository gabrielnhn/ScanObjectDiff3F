# heatmap_debug.py
import torch
import numpy as np
import trimesh
from pytorch3d.io import IO

# Paths
source_ply = "pointcloud1_with_features.ply"
source_pt = "pointcloud1_with_features.pt"
target_ply = "pointcloud2_with_features.ply"
target_pt = "pointcloud2_with_features.pt"

device = 'cuda'

print("Loading...")
f_source = torch.load(source_pt).to(device)
f_target = torch.load(target_pt).to(device)
pcd_source = trimesh.load(source_ply)
pcd_target = trimesh.load(target_ply)

# Normalize
f_source = torch.nn.functional.normalize(f_source, dim=-1)
f_target = torch.nn.functional.normalize(f_target, dim=-1)

# Pick a point index manually (e.g., point 1000) or random
# Try to pick a point on the armrest or seat
query_idx = 1000 
query_feat = f_source[query_idx].unsqueeze(0) # (1, D)

print(f"Visualizing similarity to Source Point {query_idx}...")

# Compute Cosine Sim against ALL target points
# (N_target, D) @ (D, 1) -> (N_target, 1)
sim = torch.mm(f_target, query_feat.T).squeeze().cpu().numpy()

# Normalize similarity to 0-1 for color mapping
sim = (sim - sim.min()) / (sim.max() - sim.min())

# Create Heatmap (Blue = Low, Red = High)
import matplotlib.pyplot as plt
cmap = plt.get_cmap("jet")
colors = cmap(sim)[:, :3] # (N, 3)

# Save Target with Heatmap
pcd_target.colors = (colors * 255).astype(np.uint8)
pcd_target.export("debug_heatmap_target.ply")

# Save Source with the query point highlighted (Green)
source_colors = np.zeros_like(pcd_source.vertices) + 128
source_colors[query_idx] = [0, 255, 0] # Highlight query
pcd_source.colors = source_colors.astype(np.uint8)
pcd_source.export("debug_heatmap_source.ply")

print("Done. Open 'debug_heatmap_source.ply' (look for green dot) and 'debug_heatmap_target.ply' (look for red areas).")