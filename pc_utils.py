import numpy as np
import torch
from pytorch3d.structures import Pointclouds
from pytorch3d.io import IO
import h5py
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_pc_file_with_colours(filename):
    # Load bin file
    pc = np.fromfile(filename, dtype=np.float32)

    # Reshape based on format
    # x, y, z, nx, ny, nz, r, g, b, instance_label, semantic_label
    
    pc = pc[1:].reshape((-1, 11))
    positions = np.array(pc[:, 0:3])
    normals = np.array(pc[:, 3:6]) # Capture Normals
    colours = np.array(pc[:, 6:9])/255
    # labels = np.array(pc[:, 10]).astype()
    return positions, colours, normals

def load_pc_file_with_labels(filename):
    # Load bin file
    pc = np.fromfile(filename, dtype=np.float32)

    # Reshape based on format
    # x, y, z, nx, ny, nz, r, g, b, instance_label, semantic_label
    
    pc = pc[1:].reshape((-1, 11))
    positions = np.array(pc[:, 0:3])
    normals = np.array(pc[:, 3:6]) # Capture Normals
    colours = np.array(pc[:, 6:9])/255
    labels = np.array(pc[:, 10]).astype(int)
    return positions, colours, normals, labels




def load_scanobjectnn_to_pytorch3d(filename, device, max_points=50000):
    positions_np, colours_np, normals_np, labels_np = load_pc_file_with_labels(filename)
    points_tensor = torch.from_numpy(positions_np).float().to(device)
    
    # Pack Features
    if colours_np is not None:
        # Normalize if 0-255 range detected
        if colours_np.max() > 1.0:
            colours_np = colours_np / 255.0
        features_tensor = torch.from_numpy(colours_np).float().to(device)
    else:
        features_tensor = torch.ones_like(points_tensor).to(device)

    # Pack Normals
    normals_tensor = None
    if normals_np is not None:
        normals_tensor = torch.from_numpy(normals_np).float().to(device)

    # Create PyTorch3D Structure
    from pytorch3d.structures import Pointclouds
    pcd = Pointclouds(
        points=[points_tensor], 
        features=[features_tensor],
        normals=[normals_tensor] if normals_tensor is not None else None
    )
    
    return pcd, labels_np


def save_pointcloud_with_features(pcd, filename, features=None, labels=None):
    # print("FEATURES UNSQUEEZE SHAPE", features.unsqueeze(0).shape)
    if isinstance(pcd, Pointclouds):
        points = pcd.points_padded().to(device)
    else:
        points = pcd.to(device)
    if points.dim() == 2:
        points = points.unsqueeze(0)
    
    if features is not None:
        features = features.to(device)
        if features.dim() == 2:
            features = features.unsqueeze(0)
            
        pcd_copy = Pointclouds(points=points, features=features)
    else:
        pcd_copy = Pointclouds(points=points)
    
    if not filename.endswith(".ply"):
        filename = filename.split(".")[0]
        filename = filename + ".ply"
    IO().save_pointcloud(data=pcd_copy, path=filename)
        
    if features is not None:
        if not filename.endswith(".pt"):
            filename = filename.split(".")[0]
            filename = filename + ".pt"
            
        torch.save(features, filename)

    if labels is not None:
        if not filename.endswith(".npy"):
            filename = filename.split(".")[0]
            filename = filename + ".npy"
    
        np.save(filename, labels)    
        
        
def load_mvp_to_pytorch3d(h5_filename, index=0, load_complete=False):
    with h5py.File(h5_filename, 'r') as f:
        # Usually they are 'incomplete_pcds', 'complete_pcds', and 'labels'
        
        if load_complete:
            key = 'complete_pcds' if 'complete_pcds' in f.keys() else 'gt_pcds'
            # Note: Complete PCDs are often fewer (e.g., 1 per object instead of 26 per view)
            # So the index might need wrapping depending on how MVP formatted the test set.
            # Usually, they align them 1-to-1 in the CP file to make life easy.
            idx_to_use = index if len(f[key]) == len(f['incomplete_pcds']) else index // 26
        else:
            key = 'incomplete_pcds' if 'incomplete_pcds' in f.keys() else 'partial_pcds'
            idx_to_use = index
            
        # Extract the XYZ coordinates
        points_np = f[key][idx_to_use] # Shape: (2048, 3) or (16384, 3)
        
    # Move to PyTorch
    points_tensor = torch.from_numpy(points_np).float().to(device)
    
    # MVP only provides geometry (XYZ), no RGB. 
    # We create a dummy feature tensor of ones (White color) so PyTorch3D doesn't break!
    features_tensor = torch.ones_like(points_tensor).to(device) 
    
    # Pack into PyTorch3D format
    pcd = Pointclouds(
        points=[points_tensor], 
        features=[features_tensor],
    )
    
    return pcd #, labels_np

def load_ply_to_pytorch3d(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find PLY file at: {filepath}")

    # PyTorch3D's native IO handles PLY parsing beautifully
    raw_pcd = IO().load_pointcloud(filepath, device=device)
    
    points = raw_pcd.points_padded()
    features = raw_pcd.features_padded()
    normals = raw_pcd.normals_padded()
    
    # Safety net: If the PLY lacks color data, fake it so the renderer doesn't crash
    if features is None:
        features = torch.ones_like(points).to(device)
        
    # Re-pack it safely
    safe_pcd = Pointclouds(
        points=points,
        features=features,
        normals=normals if normals is not None else None
    )
    
    return safe_pcd