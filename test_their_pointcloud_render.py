import torch
import numpy as np
from PIL import Image
import math
import sys

# --- 1. THE CODE YOU PASTED (Use exact logic) ---
from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor
)

def get_colored_depth_maps(raw_depths, H, W):
    import matplotlib.cm as cm
    import matplotlib.colors
    cmap = cm.get_cmap('Greys')
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    depth_images = []
    for i in range(raw_depths.size()[0]):
        d = raw_depths[i, ..., 0] # Fix: slice the last dim (H,W,1) -> (H,W)
        # Quick fix for background (usually -1 or very large)
        # Map background to far plane or 0 for visualization
        mask = d >= 0
        if mask.sum() > 0:
            dmin, dmax = d[mask].min(), d[mask].max()
            d = (d - dmin) / (dmax - dmin + 1e-6)
            d[~mask] = 0.0 # Background
        else:
            d[:] = 0.0
            
        flat_d = d.view(1, -1).cpu().detach().numpy()
        flat_colors = mapper.to_rgba(flat_d)
        depth_colors = np.reshape(flat_colors, (H, W, 4))[:, :, :3]
        np_image = (depth_colors * 255).astype('uint8')
        depth_images.append(np_image)
    return depth_images

def run_rendering(device, points, colors, num_views, H, W, return_images=False):
    # NOTE: The snippet provided HARDCODES .cuda() here:
    # pointclouds = Pointclouds(points=[points], features=[torch.ones(points.size()).float().to(device)])
    pointclouds = Pointclouds(points=[points], features=[colors])
    
    bbox = pointclouds.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bbox_center = (bbox_min + bbox_max) / 2.0
    
    # Distance logic
    scaling_factor = 0.65
    bb_diff = bbox_max - bbox_min
    distance = torch.sqrt((bb_diff * bb_diff).sum()) * scaling_factor
    
    steps = int(math.sqrt(num_views))
    end = 360 - 360/steps
    elevation = torch.linspace(0, end, steps).repeat(steps)
    azimuth = torch.linspace(0, end, steps).repeat_interleave(steps)
    
    rotation, translation = look_at_view_transform(
        dist=distance, azim=azimuth, elev=elevation, device=device, at=bbox_center.unsqueeze(0)
    )
    camera = PerspectiveCameras(R=rotation, T=translation, device=device)

    rasterization_settings = PointsRasterizationSettings(
        image_size=(H, W), radius=0.015, points_per_pixel=1
    )
    rasterizer = PointsRasterizer(cameras=camera, raster_settings=rasterization_settings)
    
    # IMPORTANT: The snippet provided extends the cloud but ONLY runs rasterizer (get depth)
    # It does NOT run the renderer to get RGB images.
    batch_points = pointclouds.extend(rotation.shape[0])
    fragments = rasterizer(batch_points)
    raw_depth = fragments.zbuf

    if return_images:
        list_depth_images_np = get_colored_depth_maps(raw_depth, H, W)
        return list_depth_images_np
    return None

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
        colours = np.array(pc[:, 6:9])
        return positions, colours

# --- 3. RUN IT ---
TEST_FILE = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/pillow/014_00015.bin"
device = torch.device('cuda:0')

# Load and Prep
xyz_np, rgb_np = load_pc_file_with_colours(TEST_FILE)
points_tensor = torch.from_numpy(xyz_np).float().to(device)
colors_tensor = torch.from_numpy(rgb_np).float().to(device)

# Run Render
print(f"Rendering {len(points_tensor)} points...")
images = run_rendering(device, points_tensor, colors_tensor, num_views=4, H=512, W=512, return_images=True)

# Save
for i, img in enumerate(images):
    Image.fromarray(img).save(f"test_render_depth_{i}.png")
    print(f"Saved test_render_depth_{i}.png")