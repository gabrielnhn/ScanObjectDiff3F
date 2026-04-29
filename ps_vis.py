path=f"/home/gabrielnhn/datasets/synthetic_redwood/upload/plyobj/indata/stanford-bunny.ply"

import polyscope as ps
import open3d as o3d
import numpy as np

# 1. Load the PLY file using Open3D
pcd = o3d.io.read_point_cloud(path)

# 2. Extract point positions as a numpy array
points = np.asarray(pcd.points)

# 3. Initialize Polyscope
ps.init()

# 4. Register the point cloud
ps.register_point_cloud("my ply cloud", points, material="normal")

# 5. Show
ps.show()
