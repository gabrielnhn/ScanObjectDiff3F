import numpy as np
import torch
from pytorch3d.structures import Pointclouds
from pytorch3d.io import IO

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




def load_scanobjectnn_to_pytorch3d(filename, device):
    positions_np, colours_np, normals_np, labels_np = load_pc_file_with_labels(filename)

    # if colours_np is not None and colours_np.max() > 1.0:
    #     colours_np = colours_np / 255.0

    # if colours_np is not None and colours_np.max() < 2:
    # colours_np = colours_np / 255



    points_tensor = torch.from_numpy(positions_np).float().to(device)
    
    # Pack Features (Color)
    features_tensor = None
    if colours_np is not None:
        features_tensor = torch.from_numpy(colours_np).float().to(device)
    else:
        features_tensor = torch.ones_like(points_tensor).to(device) # Default white

    # Pack Normals
    normals_tensor = None
    if normals_np is not None:
        normals_tensor = torch.from_numpy(normals_np).float().to(device)

    # Create Pointclouds object with BOTH features and normals
    pcd = Pointclouds(
        points=[points_tensor], 
        features=[features_tensor],
        normals=[normals_tensor] if normals_tensor is not None else None
    )
    
    return pcd, labels_np


def save_pointcloud_with_features(pcd, features, filename, labels=None):
    # print("FEATURES UNSQUEEZE SHAPE", features.unsqueeze(0).shape)
    points = pcd.points_padded().to(device)
    # features = features.unsqueeze(0).to(device)
    pcd_copy = Pointclouds(points=points
                        #    , features=features
                           )
    
    if not filename.endswith(".ply"):
        filename = filename.split(".")[0]
        filename = filename + ".ply"
        
    IO().save_pointcloud(data=pcd_copy, path=filename)
    if not filename.endswith(".pt"):
        filename = filename.split(".")[0]
        filename = filename + ".pt"
        
    torch.save(features, filename)

    if labels is not None:
        if not filename.endswith(".npy"):
            filename = filename.split(".")[0]
            filename = filename + ".npy"
    
        np.save(filename, labels)    