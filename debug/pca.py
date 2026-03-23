import torch
import numpy as np
import trimesh
from sklearn.decomposition import PCA
from pytorch3d.io import IO

def load_data(ply_path, feat_path):
    print(f"\nLoading features from {feat_path}...")
    try:
        features = torch.load(feat_path, map_location='cpu', weights_only=True).squeeze()
    except FileNotFoundError:
        print(f"Feature file {feat_path} not found.")
        return None, None

    print(f"Loading geometry from {ply_path}...")
    try:
        mesh = trimesh.load(ply_path)
        vertices = mesh.vertices
    except:
        pcd = IO().load_pointcloud(ply_path)
        vertices = pcd.points_padded()[0].numpy()

    if len(vertices) != len(features):
        print(f"Shape Mismatch! Vertices: {len(vertices)}, Features: {len(features)}")
        min_len = min(len(vertices), len(features))
        vertices = vertices[:min_len]
        features = features[:min_len]
        
    return vertices, features

def compute_global_pca_colors(features_list):
    """
    Takes a list of feature tensors, concatenates them to fit a global PCA,
    and returns a list of RGB color arrays split back to their original sizes.
    """
    # 1. Record original lengths and concatenate
    lengths = [len(f) for f in features_list]
    combined_features = torch.cat(features_list, dim=0)
    
    # 2. Safety Checks
    if torch.isnan(combined_features).any() or torch.isinf(combined_features).any():
        print("!!! CRITICAL FAILURE: Features contain NaNs or Infs !!!")
        return [np.zeros((l, 3)) for l in lengths]

    std_dev = torch.std(combined_features, dim=0).mean().item()
    print(f"Global Feature Standard Deviation: {std_dev:.6f}")
    
    if std_dev < 1e-4:
        print("!!! CRITICAL FAILURE: Features are constant. DINO saw nothing. !!!")
        return [np.zeros((l, 3)) for l in lengths]
    
    # 3. Run Global PCA
    X = combined_features.cpu().numpy()
    print(f"Running Global PCA on {X.shape[0]} total points...")
    pca = PCA(n_components=3)
    components = pca.fit_transform(X)
    
    # 4. Global Min-Max Normalization
    # It is crucial to normalize using the global min/max so colors match across PCs
    for i in range(3):
        col = components[:, i]
        col_min, col_max = col.min(), col.max()
        # Add 1e-8 to prevent division by zero
        components[:, i] = (col - col_min) / (col_max - col_min + 1e-8)
        
    global_colors = (components * 255).astype(np.uint8)
    
    # 5. Split colors back into the respective point clouds
    split_colors = []
    start_idx = 0
    for l in lengths:
        split_colors.append(global_colors[start_idx : start_idx + l])
        start_idx += l
        
    return split_colors

def main():
    # Define your files here
    files = [
        {"ply": "pointcloud1_with_features.ply", "feat": "pointcloud1_with_features.pt", "out": "1"},
        {"ply": "pointcloud2_with_features.ply", "feat": "pointcloud2_with_features.pt", "out": "2"}
    ]
    
    all_vertices = []
    all_features = []
    valid_files = []
    
    # Load everything
    for f in files:
        vertices, features = load_data(f["ply"], f["feat"])
        if vertices is not None and features is not None:
            all_vertices.append(vertices)
            all_features.append(features)
            valid_files.append(f)
            
    if not all_features:
        print("No valid data loaded. Exiting.")
        return

    # Compute shared PCA colors
    pca_colors_list = compute_global_pca_colors(all_features)
    
    # Save Outputs
    print("\nSaving colored point clouds...")
    for f, vertices, colors in zip(valid_files, all_vertices, pca_colors_list):
        out_name = f"debug_pca_visual_shared_{f['out']}.ply"
        pcd_out = trimesh.PointCloud(vertices=vertices, colors=colors)
        pcd_out.export(out_name)
        print(f"Saved -> {out_name}")

if __name__ == "__main__":
    main()