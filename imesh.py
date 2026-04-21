import os
import sys
import gc
import torch
import numpy as np
from PIL import Image
import rembg
from einops import rearrange
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

# TRICK PYTHON: Fake nvdiffrast to bypass rasterizer
from unittest.mock import MagicMock
sys.modules['nvdiffrast'] = MagicMock()
sys.modules['nvdiffrast.torch'] = MagicMock()

sys.path.append("./instantmesh")
import models.lrm_mesh
from utils.camera_util import get_zero123plus_input_cameras

device = "cuda"
model_cache_dir = './ckpts/'
os.makedirs(model_cache_dir, exist_ok=True)

# ==========================================
# PHASE 1: IMAGE PREP (THE WHITE BACKGROUND)
# ==========================================
print("1. Processing Input Image...")
input_image_path = "manual-bunny.png"


# Remove background
raw_img = Image.open(input_image_path)
no_bg_img = rembg.remove(raw_img)

# Paste onto a PURE WHITE background (CRITICAL FOR INSTANTMESH)
white_bg = Image.new("RGBA", no_bg_img.size, "WHITE")
white_bg.paste(no_bg_img, (0, 0), mask=no_bg_img)
processed_image = white_bg.convert("RGB")
processed_image = processed_image.resize((320, 320)) # Ensure standard size

# ==========================================
# PHASE 2: ZERO123++ (CUSTOM UNET)
# ==========================================
print("2. Loading Custom Zero123++...")
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", 
    # custom_pipeline="zero123plus",
    custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16,
    # cache_dir=model_cache_dir
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)

# Load the custom white-background UNet from InstantMesh authors
unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh",
                                 filename="diffusion_pytorch_model.bin",
                                 repo_type="model",
                                 cache_dir=model_cache_dir)

pipeline.unet.load_state_dict(torch.load(unet_ckpt_path, map_location='cpu'), strict=True)
pipeline = pipeline.to(device)

print("   Generating 6 multi-views...")
z123_image = pipeline(processed_image, num_inference_steps=50).images[0]

# --- THE AIRLOCK (SAVE VRAM) ---
print("   Flushing Zero123++ from VRAM...")
del pipeline
gc.collect()
torch.cuda.empty_cache()
# -------------------------------

# ==========================================
# PHASE 3: INSTANTMESH POINT CLOUD
# ==========================================
print("3. Formatting images for InstantMesh...")
# Convert the 960x640 grid directly into a [6, 3, 320, 320] tensor using einops (Zero cropping mistakes!)
images_arr = np.asarray(z123_image, dtype=np.float32) / 255.0
images_tensor = torch.from_numpy(images_arr).permute(2, 0, 1).contiguous()
images_tensor = rearrange(images_tensor, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)

# Batch it and cast to FP16
image_tensor = images_tensor.unsqueeze(0).to(device, dtype=torch.float16)
cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device, dtype=torch.float16)

print("4. Loading InstantMesh...")
model_ckpt_path = hf_hub_download(
    repo_id="TencentARC/InstantMesh",
    filename="instant_mesh_base.ckpt",
    repo_type="model",
    cache_dir=model_cache_dir
)
# grid_res=64 prevents the SDF OOM crash!
model = models.lrm_mesh.InstantMesh(grid_res=64)
state_dict = torch.load(model_ckpt_path, map_location='cpu', weights_only=True)['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.') and 'source_camera' not in k}
model.load_state_dict(state_dict, strict=True)

model = model.to(device, dtype=torch.float16)
model.init_flexicubes_geometry(device)

print("   Extracting 3D Geometry...")
with torch.no_grad():
    planes = model.forward_planes(image_tensor, cameras)
    
    # Flush memory right before the FlexiCubes SDF step
    torch.cuda.empty_cache()
    
    # (Note: make sure you kept your .float() casting fix inside flexicubes_geometry.py!)
    mesh_v, mesh_f, _, _, _, _ = model.get_geometry_prediction(planes)
    vertices = mesh_v[0]

# ==========================================
# PHASE 4: SAVE PLY
# ==========================================
points = vertices.detach().cpu().numpy()

def save_point_cloud_to_ply(points, filename):
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

output_filename = "final_instantmesh_shape.ply"
save_point_cloud_to_ply(points, output_filename)
print(f"Success! Point cloud saved to {output_filename}")