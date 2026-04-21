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
    def __init__(self, background_color=(1.0, 1.0, 1.0), ambient=0.6, diffuse=0.8):
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
    """ 
    Render point clouds using their normal vectors as RGB colors. 
    Follows the ComPC normal-to-color mapping strategy.
    """
    def __init__(self, background_color=(1.0, 1.0, 1.0)):
        super().__init__()
        # You can change the default background to (0.5, 0.5, 0.5) if you want 
        # the background to represent a perfectly flat/neutral normal.
        self.compositor = AlphaCompositor(background_color=background_color)

    def forward(self, fragments, pcd_batch, cameras=None):
        weights = (fragments.idx != -1).float().permute(0, 3, 1, 2)
        indices = fragments.idx.long().permute(0, 3, 1, 2)
        
        normals = pcd_batch.normals_packed()
        
        if normals is None:
            raise ValueError("Point cloud must contain normals to render normal maps.")

        # ComPC Strategy: Scale normals from [-1.0, 1.0] to [0.0, 1.0] for RGB
        scaled_normals = (normals + 1.0) / 2.0
        
        # Clamp to ensure strict RGB bounds (handles minor floating point errors)
        scaled_normals = torch.clamp(scaled_normals, 0.0, 1.0)
        
        # PyTorch3D compositor expects features in shape (Channels, Points)
        scaled_normals = scaled_normals.permute(1, 0)

        images = self.compositor(indices, weights, scaled_normals)
        
        # Return in standard (Batch, Height, Width, Channels) format
        return images.permute(0, 2, 3, 1)