import torch
import numpy as np
import trimesh
from pytorch3d.io import IO
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

source_ply_path = "pointcloud1_with_features.ply"
source_feat_path = "pointcloud1_with_features.pt"

target_ply_path = "pointcloud2_with_features.ply"
target_feat_path = "pointcloud2_with_features.pt"


def save_ply_with_colors(points, colors, filename):
    # Ensure colors are 0-255 uint8
    if colors.max() <= 1.0 and colors.max() > 0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)
        
    pcd = trimesh.PointCloud(vertices=points, colors=colors)
    pcd.export(filename)
    print(f"Saved {filename}")

def run_transfer():
    print("1. Loading Data...")
    
    f_source = torch.load(source_feat_path).to(device).squeeze()
    f_target = torch.load(target_feat_path).to(device).squeeze()
    
    pcd_source = IO().load_pointcloud(source_ply_path).to(device)
    pcd_target = IO().load_pointcloud(target_ply_path).to(device)
    
    verts_source = pcd_source.points_padded()[0]
    verts_target = pcd_target.points_padded()[0]
    
    labels_source = np.load("pointcloud1_with_features.npy")
    
    # Sanity Check
    if len(labels_source) != f_source.shape[0]:
        print(f"WARNING: Label count {len(labels_source)} != Feature count {f_source.shape[0]}")
        # If mismatch, force trim to min length (rare, but safer)
        min_len = min(len(labels_source), f_source.shape[0])
        labels_source = labels_source[:min_len]
        f_source = f_source[:min_len]
        verts_source = verts_source[:min_len]

    print("2. Filtering Source Features (Ignoring Grey/Background)...")
    
    # valid_mask = labels_source != -1
    valid_mask = labels_source
    
    f_source_clean = f_source[valid_mask]
    labels_source_clean = labels_source[valid_mask]
    
    
    print(f"   Original Source Points: {len(labels_source)}")
    print(f"   Valid Part Points:      {len(labels_source_clean)} (Background removed)")

    print("3. Calculating Correspondence...")
    
    # Normalize features for Cosine Similarity
    f_source_norm = torch.nn.functional.normalize(f_source_clean, dim=-1)
    f_target_norm = torch.nn.functional.normalize(f_target, dim=-1)
    
    # Matrix Multiplication: (N_target, D) @ (D, N_source_clean) -> (N_target, N_source_clean)
    sim_matrix = torch.mm(f_target_norm, f_source_norm.T)
    
    # Find best match
    best_match_indices = torch.argmax(sim_matrix, dim=1).cpu().numpy()
    
    # --- TRANSFER LABELS ---
    # Target Point[i] gets label of Clean Source Point[best_match[i]]
    predicted_labels = labels_source_clean[best_match_indices]
    
    print("4. Saving Visualizations...")
    
    # Robust Palette (Red, Green, Blue, Yellow, Cyan, Magenta)
    palette = np.array([
        [255, 0, 0],   # Class 0
        [0, 255, 0],   # Class 1
        [0, 0, 255],   # Class 2
        [255, 255, 0], # Class 3
        [0, 255, 255], # Class 4
        [255, 0, 255], # Class 5
    ])
    
    # Helper to map labels to colors safely
    def map_colors(lbls):
        # Clamp labels to palette size to prevent crash
        safe_lbls = np.clip(lbls, 0, len(palette)-1)
        return palette[safe_lbls]

    # 1. Save Target Prediction
    target_colors = map_colors(predicted_labels)
    save_ply_with_colors(verts_target.cpu().numpy(), target_colors, "final_transfer_result.ply")
    
    # 2. Save Source GT (just to compare)
    # We map -1 (background) to Grey [128, 128, 128] manually for this visual
    source_colors = np.zeros((len(labels_source), 3), dtype=np.uint8) + 128
    for cls_id in np.unique(labels_source):
        if cls_id >= 0:
            mask = labels_source == cls_id
            source_colors[mask] = palette[cls_id % len(palette)]
            
    save_ply_with_colors(verts_source.cpu().numpy(), source_colors, "final_source_gt.ply")

    print("\nDONE.")
    print("-> 'final_transfer_result.ply' should now have NO greys (full transfer).")
    print("-> 'final_source_gt.ply' shows the original with greys.")

if __name__ == "__main__":
    run_transfer()