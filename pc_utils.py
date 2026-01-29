import numpy as np
import torch
from pytorch3d.structures import Pointclouds


# --- 1. Update Loader to get Normals ---
def load_pc_file_with_colours(filename, suncg=False):
    # Load bin file
    pc = np.fromfile(filename, dtype=np.float32)

    # Reshape based on format
    # x, y, z, nx, ny, nz, r, g, b, ...
    if suncg:
        pc = pc[1:].reshape((-1, 3))
        positions = pc[:, 0:3]
        return positions, None, None
    else:
        pc = pc[1:].reshape((-1, 11))
        positions = np.array(pc[:, 0:3])
        normals = np.array(pc[:, 3:6]) # Capture Normals
        colours = np.array(pc[:, 6:9])/255
        return positions, colours, normals


def load_scanobjectnn_to_pytorch3d(filename, device):
    positions_np, colours_np, normals_np = load_pc_file_with_colours(filename)

    # Normalize colors to [0, 1]
    if colours_np is not None and colours_np.max() > 1.0:
        colours_np = colours_np / 255.0

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
    
    return pcd


