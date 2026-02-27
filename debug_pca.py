import torch
import numpy as np
import trimesh
from sklearn.decomposition import PCA
from pytorch3d.io import IO
import matplotlib.pyplot as plt

# --- CONFIG ---
# Use the file you successfully processed (or the one giving gray heatmaps)
# PLY_PATH = "pointcloud1_with_features.ply" 
# FEAT_PATH = "pointcloud1_with_features.pt"
PLY_PATH = "pointcloud2_with_features.ply" 
FEAT_PATH = "pointcloud2_with_features.pt"

def compute_pca_colors(features):
    """
    Compresses 768-dim DINO features into 3-dim RGB using PCA.
    """
    # 1. Check for NaNs or Infinite values
    if torch.isnan(features).any() or torch.isinf(features).any():
        print("!!! CRITICAL FAILURE: Features contain NaNs or Infs !!!")
        return np.zeros((len(features), 3))

    # 2. Check Variance (The "Gray Heatmap" Check)
    # Calculate std dev per dimension, then average
    std_dev = torch.std(features, dim=0).mean().item()
    print(f"Feature Standard Deviation: {std_dev:.6f}")
    
    if std_dev < 1e-4:
        print("!!! CRITICAL FAILURE: Features are constant (Zero Variance). DINO saw nothing. !!!")
        return np.zeros((len(features), 3))
    
    # 3. Move to CPU for sklearn
    X = features.cpu().numpy()
    
    # 4. Run PCA (Reduce to 3 components for RGB)
    print("Running PCA...")
    pca = PCA(n_components=3)
    components = pca.fit_transform(X)
    
    # 5. Normalize to [0, 255] for coloring
    # Min-Max normalization per channel
    for i in range(3):
        col = components[:, i]
        col_min, col_max = col.min(), col.max()
        components[:, i] = (col - col_min) / (col_max - col_min)
        
    return (components * 255).astype(np.uint8)

def main():
    print(f"Loading {FEAT_PATH}...")
    try:
        features = torch.load(FEAT_PATH, map_location='cpu').squeeze()
    except FileNotFoundError:
        print("Feature file not found. Did the previous script crash before saving?")
        return

    print(f"Loading Geometry from {PLY_PATH}...")
    try:
        # Try loading with Trimesh first as it's more robust for generic PLYs
        mesh = trimesh.load(PLY_PATH)
        vertices = mesh.vertices
    except:
        # Fallback to PyTorch3D
        pcd = IO().load_pointcloud(PLY_PATH)
        vertices = pcd.points_padded()[0].numpy()

    if len(vertices) != len(features):
        print(f"Shape Mismatch! Vertices: {len(vertices)}, Features: {len(features)}")
        # If mismatch is small, truncate to common length
        min_len = min(len(vertices), len(features))
        vertices = vertices[:min_len]
        features = features[:min_len]

    # Compute Colors
    pca_colors = compute_pca_colors(features)
    
    # Save Output
    out_name = "debug_pca_visual.ply"
    print(f"Saving PCA visualization to {out_name}...")
    
    pcd_out = trimesh.PointCloud(vertices=vertices, colors=pca_colors)
    pcd_out.export(out_name)
    print("Done. Open 'debug_pca_visual.ply' in MeshLab/CloudCompare.")

if __name__ == "__main__":
    main()