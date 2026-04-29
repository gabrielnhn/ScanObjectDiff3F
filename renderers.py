import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.renderer import (
    look_at_view_transform,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
    FoVPerspectiveCameras
)
from pytorch3d.ops import knn_points


class PhongCircleRenderer(nn.Module):
    """ Render circles with Blinn-Phong shading. """
    def __init__(self, background_color=(1.0, 1.0, 1.0), ambient=0.1, diffuse=0.7):
        super().__init__()
        self.compositor = AlphaCompositor(background_color=background_color)
        self.ambient = ambient
        self.diffuse = diffuse

    def forward(self, fragments, pcd_batch, cameras=None, light_dir=torch.tensor([0.0, 1.0, 1.0])):
        weights = (fragments.idx != -1).float().permute(0, 3, 1, 2)
        indices = fragments.idx.long().permute(0, 3, 1, 2)
        
        points = pcd_batch.points_packed()
        features = pcd_batch.features_packed()
        normals = pcd_batch.normals_packed()
        
        if normals is None:
            raise ValueError("need normals")

        light_dir = F.normalize(light_dir.to(points.device), p=2, dim=-1)
        n_dot_l = torch.sum(normals * light_dir, dim=-1, keepdim=True)
        n_dot_l = torch.where(n_dot_l < 0, -n_dot_l, n_dot_l)
                
        diffuse_term = torch.clamp(n_dot_l, min=0.0)
        shaded_features = features * (self.ambient + self.diffuse * diffuse_term)
        shaded_features = torch.clamp(shaded_features, 0.0, 1.0)
        shaded_features = shaded_features.permute(1, 0)

        images = self.compositor(indices, weights, shaded_features)
        return images.permute(0, 2, 3, 1)


class NormalsRenderer(nn.Module):
    def __init__(self, background_color=(0.5, 0.5, 0.5), cameras=None):
        super().__init__()
        # Using the standard background color
        self.compositor = AlphaCompositor(background_color=background_color)
        self.cameras = cameras

    def forward(self, fragments, pcd_batch):
        weights = (fragments.idx != -1).float().permute(0, 3, 1, 2)
        indices = fragments.idx.long().permute(0, 3, 1, 2)
        
        # Get world-space points and normals
        points = pcd_batch.points_packed()   # (P, 3)
        normals = pcd_batch.normals_packed() # (P, 3)
        
        # 1. Get the camera center in world coordinates. 
        # This corresponds exactly to Open3D's `pose[:3, 3]`.
        # We take [0:1] to ensure shape (1, 3) for clean broadcasting, matching your batch=1 setup.
        cam_loc = self.cameras.get_camera_center()[0:1] 
        
        # 2. Orient normals towards the camera (Matches `orient_normals_towards_camera_location`)
        # Calculate the view direction vector from each point to the camera
        view_dirs = cam_loc - points 
        
        # Check if normals are pointing away from the camera using a dot product
        dot_products = (normals * view_dirs).sum(dim=-1, keepdim=True) # (P, 1)
        
        # Flip normals that are pointing away (where dot product is negative)
        oriented_normals = torch.where(dot_products < 0, -normals, normals)
        
        # 3. Map to [0, 1] color range (Matches `norm_normal: (normal+1)/2`)
        colors = (oriented_normals + 1.0) / 2.0
        
        # Clamp just to be safe against minor floating point precision overshoots
        colors = torch.clamp(colors, 0.0, 1.0)
        
        # 4. Composite the image
        # Using the same permute(1, 0) you used to fit your specific compositor shape (3, P)
        images = self.compositor(indices, weights, colors.permute(1, 0))
        
        return images.permute(0, 2, 3, 1)