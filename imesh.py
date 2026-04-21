from huggingface_hub import hf_hub_download
import torch
import os
import sys
sys.path.append("./instantmesh")
import models.lrm_mesh

from utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)

device="cuda"


# Define the cache directory for model files
model_cache_dir = './ckpts/'
os.makedirs(model_cache_dir, exist_ok=True)


# load reconstruction model
print('Loading reconstruction model ...')
model_ckpt_path = hf_hub_download(
    repo_id="TencentARC/InstantMesh",
    filename="instant_mesh_base.ckpt",
    repo_type="model",
    cache_dir=model_cache_dir
)


model = models.lrm_mesh.InstantMesh()

state_dict = torch.load(model_ckpt_path, map_location='cpu', weights_only=True)['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.') and 'source_camera' not in k}
model.load_state_dict(state_dict, strict=True)

# Move to GPU and cast to Half-Precision (Cuts model memory in half)
model = model.to(device, dtype=torch.float16)


cameras = get_zero123plus_input_cameras()

import torchvision.transforms.functional as TF

from PIL import Image
images_pil = Image.open("COLORS_GRID_post.png").convert("RGB")
width, height = images_pil.size
# Zero123++ grid is 3 rows, 2 columns (or 2x3 depending on how you saved it)
# Assuming a 2-column, 3-row grid like the default Zero123++ output:
tile_w = width // 2
tile_h = height // 3

sliced_views = []
for y in range(3):
    for x in range(2):
        # Crop each individual view out of the grid
        box = (x * tile_w, y * tile_h, (x + 1) * tile_w, (y + 1) * tile_h)
        view = images_pil.crop(box)
        # Resize to InstantMesh's expected resolution (usually 320x320)
        view = view.resize((320, 320), Image.Resampling.LANCZOS)
        # Convert to tensor [C, H, W] in range [0, 1]
        tensor_view = TF.to_tensor(view)
        sliced_views.append(tensor_view)

# Stack into [Views, C, H, W]
image_tensor = torch.stack(sliced_views)

# Add Batch dimension: [Batch, Views, C, H, W] -> [1, 6, 3, 320, 320]
image_tensor = image_tensor.unsqueeze(0).to(device, dtype=torch.float16)

# Ensure cameras are also batched [1, 6, 16] and on the right device
cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device, dtype=torch.float16)

# NOW you can safely pass it to the model!

with torch.inference_mode():
    planes = model.forward_planes(image_tensor, cameras)
    mesh_v, mesh_f, _, _, _, _ = model.get_geometry_prediction(planes)
    vertices = mesh_v[0] # Your point cloud!

points = vertices.detach().cpu().numpy()

# 2. Define a fast, native ASCII PLY writer (No extra libraries needed!)
def save_point_cloud_to_ply(points, filename):
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        # Write all points
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

# 3. Save the file!
output_filename = "generated_shape_instantmesh.ply"
save_point_cloud_to_ply(points, output_filename)
print(f"Success! Point cloud saved to {output_filename}")