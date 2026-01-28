from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras
import torch
import math
import numpy as np
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor)

def load_pc_file_with_colours(filename, suncg = False, with_bg = True):
    #load bin file
    pc=np.fromfile(filename, dtype=np.float32)
    # pc=np.fromfile(os.path.join(DATA_PATH, filename), dtype=np.float32)

    #first entry is the number of points
    #then x, y, z, nx, ny, nz, r, g, b, label, nyu_label
    if(suncg):
        pc = pc[1:].reshape((-1,3))
    else:
        pc = pc[1:].reshape((-1,11))

    positions = np.array(pc[:,0:3])
    colours = np.array(pc[:,6:9])
    return positions, colours


def load_scanobjectnn_to_pytorch3d(filename, device):
    positions_np, colours_np = load_pc_file_with_colours(filename)

    if colours_np.max() > 1.0:
        colours_np = colours_np / 255.0

    points_tensor = torch.from_numpy(positions_np).float().to(device)
    features_tensor = torch.from_numpy(colours_np).float().to(device)

    pcd = Pointclouds(points=[points_tensor], features=[features_tensor])
    
    return pcd


def get_colored_depth_maps(raw_depths,H,W):
    import matplotlib
    import matplotlib.cm as cm
    cmap = cm.get_cmap('Greys')
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    depth_images = []
    for i in range(raw_depths.size()[0]):
        d = raw_depths[i]
        dmax = torch.max(d) ; dmin = torch.min(d)
        d = (d-dmin)/(dmax-dmin)
        flat_d = d.view(1,-1).cpu().detach().numpy()
        flat_colors = mapper.to_rgba(flat_d)
        depth_colors = np.reshape(flat_colors,(H,W,4))[:,:,:3]
        np_image = depth_colors*255
        np_image = np_image.astype('uint8')
        depth_images.append(np_image)

    return depth_images


@torch.no_grad()
def run_rendering(device, points, num_views, H, W, add_angle_azi=0, add_angle_ele=0, use_normal_map=False,return_images=False):
    # FIX 1: Use .to(device) instead of hardcoded .cuda() for safety
    # FIX 2: Create features using ones_like to match input tensor properties
    pointclouds = Pointclouds(points=[points], features=[torch.ones_like(points).float().to(device)])
    
    bbox = pointclouds.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bb_diff = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) / 2.0
    scaling_factor = 0.65
    distance = torch.sqrt((bb_diff * bb_diff).sum())
    distance *= scaling_factor
    
    # FIX 3: Safety check for steps
    steps = int(math.sqrt(num_views))
    if steps < 1: steps = 1
        
    end = 360 - 360/steps
    elevation = torch.linspace(start = 0 , end = end , steps = steps).repeat(steps) + add_angle_ele
    azimuth = torch.linspace(start = 0 , end = end , steps = steps)
    azimuth = torch.repeat_interleave(azimuth, steps) + add_angle_azi
    bbox_center = bbox_center.unsqueeze(0)
    rotation, translation = look_at_view_transform(
        dist=distance, azim=azimuth, elev=elevation, device=device, at=bbox_center
    )
    camera = PerspectiveCameras(R=rotation, T=translation, device=device)

    #rasterizer
    rasterization_settings = PointsRasterizationSettings(
        image_size=H,
        radius = 0.01,
        points_per_pixel = 1,
        bin_size = 0,
        max_points_per_bin = 0
    )

    #render pipeline
    rasterizer = PointsRasterizer(cameras=camera, raster_settings=rasterization_settings)
    camera_centre = camera.get_camera_center()
    batch_renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )
    
    # FIX 4: Use actual view count derived from rotation matrix
    actual_views = rotation.shape[0]
    batch_points = pointclouds.extend(actual_views)
    
    fragments = rasterizer(batch_points)
    raw_depth = fragments.zbuf

    if not return_images:
        return None,None,camera,raw_depth
    else:
        list_depth_images_np = get_colored_depth_maps(raw_depth,H,W)
        return None,None,camera,raw_depth,list_depth_images_np


def batch_render(device, points, num_views, H, W, use_normal_map=False,return_images=False):
    trials = 0
    add_angle_azi = 0
    add_angle_ele = 0
    while trials < 5:
        try:
            return run_rendering(device, points, num_views, H, W, add_angle_azi=add_angle_azi, add_angle_ele=add_angle_ele, use_normal_map=use_normal_map,return_images=return_images)
        except torch.linalg.LinAlgError as e:
            trials += 1
            print("lin alg exception at rendering, retrying ", trials)
            add_angle_azi = torch.randn(1)
            add_angle_ele = torch.randn(1)
            continue
        
