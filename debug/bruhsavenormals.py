from pytorch3d.structures import Pointclouds
from pytorch3d.io import IO
import os
from pytorch3d.ops import estimate_pointcloud_normals
import numpy as np
device = "cuda"
import trimesh


def BRUH(filepath, normal_factor):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find PLY file at: {filepath}")

    raw_pcd = IO().load_pointcloud(filepath, device=device)
    
    points = raw_pcd.points_padded()
    features = raw_pcd.features_padded()
    normals = raw_pcd.normals_padded()
    
    if normals is None and normal_factor:
        print("No normals found. Computing them via PyTorch3D local PCA...")
        
        k =  int(normal_factor *np.sqrt(len(points[0])))
        print(f"Computing using K={k} neighbours")
        normals = estimate_pointcloud_normals(
            points, 
            neighborhood_size=k,
            disambiguate_directions=True
        )
        
    safe_pcd = Pointclouds(
        points=points,
        features=normals
    )
    
    IO().save_pointcloud(safe_pcd, "debug/withnormals.ply")
    trimesh.load("debug/withnormals.ply").show()
    
if __name__ == "__main__":
    BRUH(
            # f"/home/gabrielnhn/datasets/synthetic_redwood/upload/plyobj/indata/horse.ply",
            f"/home/gabrielnhn/datasets/synthetic_redwood/upload/plyobj/indata/stanford-bunny.ply",
            normal_factor=7
        )
    