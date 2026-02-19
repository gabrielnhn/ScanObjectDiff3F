import torch
import numpy as np
import trimesh
from pytorch3d.io import IO
# import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

source_ply_path = "pointcloud1_with_features.ply"
source_feat_path = "pointcloud1_with_features.pt"
source_lbl_path  = "pointcloud1_with_features.npy"

target_ply_path = "pointcloud2_with_features.ply"
target_feat_path = "pointcloud2_with_features.pt"
target_lbl_path = "pointcloud2_with_features.npy"

def save_ply_with_colors(points, colors, filename):
    if colors.max() <= 1.0 and colors.max() > 0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)
        
    pcd = trimesh.PointCloud(vertices=points, colors=colors)
    pcd.export(filename)
    print(f"Saved {filename}")

def find_correspondences_chunked(feat_source, feat_target, chunk_size=2000):
    """
    Computes argmax(feat_target @ feat_source.T) in blocks.
    Never allocates the full NxM matrix.
    source: Gemini
    """
    n_target = feat_target.shape[0]
    # n_source = feat_source.shape[0]
    
    nearest_indices = torch.zeros(n_target, dtype=torch.long, device='cpu')
    feat_source_t = feat_source.T
    
    for i in range(0, n_target, chunk_size):
        end = min(i + chunk_size, n_target)
        
        target_chunk = feat_target[i:end]
        
        sim_chunk = torch.mm(target_chunk, feat_source_t)
        
        best_vals, best_idxs = torch.max(sim_chunk, dim=1)
        
        # save indices to CPU and discard the chunk
        nearest_indices[i:end] = best_idxs.cpu()
        
        # Free memory
        del sim_chunk
        del best_idxs
        
    return nearest_indices

def run_transfer():
    print("Loading Data...")
    
    # Load features to GPU directly
    f_source = torch.load(source_feat_path, map_location=device).squeeze()
    f_target = torch.load(target_feat_path, map_location=device).squeeze()
    
    # Load Geometry (for saving later)
    pcd_source = IO().load_pointcloud(source_ply_path)
    pcd_target = IO().load_pointcloud(target_ply_path)
    
    verts_source = pcd_source.points_padded()[0].numpy()
    verts_target = pcd_target.points_padded()[0].numpy()
    
    labels_source = np.load(source_lbl_path)
    labels_target = np.load(target_lbl_path)

    valid_mask = labels_source != -1
    
    f_source_clean = f_source[valid_mask]
    labels_source_clean = labels_source[valid_mask]
    
    print(f"   Original Source Points: {len(labels_source)}")
    print(f"   Valid Part Points:      {len(labels_source_clean)} (Background removed)")

    # Normalize first
    f_source_norm = torch.nn.functional.normalize(f_source_clean, dim=-1)
    f_target_norm = torch.nn.functional.normalize(f_target, dim=-1)
    
    # Run Chunked Matching
    best_match_indices = find_correspondences_chunked(f_source_norm, f_target_norm, chunk_size=5000)
    
    # --- Label Transfer ---
    print("Transferring Labels...")
    
    # best_match_indices is on CPU, labels_source_clean is numpy
    best_match_indices = best_match_indices.numpy()
    predicted_labels = labels_source_clean[best_match_indices]
    
    # print(torch.count_nonzero(torch.eq(predicted_labels, torch.tensor(labels_target)))
    #       / torch.numel(predicted_labels))
    
    equal = np.where(predicted_labels == labels_target, 1, 0)
    print(np.count_nonzero(equal), "/", np.size(equal), " == ", np.count_nonzero(equal) / np.size(equal))
    
    
    
    
    # --- Visualization ---
    palette = np.array([
        [255, 0, 0],  
        [0, 255, 0],  
        [0, 0, 255],  
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
    ])
    
    def map_colors(lbls):
        safe_lbls = np.clip(lbls, 0, len(palette)-1)
        return palette[safe_lbls]

    target_colors = map_colors(predicted_labels)
    save_ply_with_colors(verts_target, target_colors, "final_transfer_result.ply")
    
    print("\nDONE.")
    save_ply_with_colors(verts_source, map_colors(labels_source), "final_source_gt.ply") 
              

if __name__ == "__main__":
    with torch.no_grad(): # Disable gradients to save even more memory
        run_transfer()