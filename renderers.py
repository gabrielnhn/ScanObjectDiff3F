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
    def __init__(self, background_color=(0.5, 0.5, 1.0), cameras=None):
        super().__init__()
        # Use your Pyglet background color (Light Blue)
        self.compositor = AlphaCompositor(background_color=background_color)
        self.cameras = cameras

    def forward(self, fragments, pcd_batch):
        weights = (fragments.idx != -1).float().permute(0, 3, 1, 2)
        indices = fragments.idx.long().permute(0, 3, 1, 2)
        
        # 1. Get World Space Normals
        normals = pcd_batch.normals_packed()  # (P, 3)
        
        # 2. Convert to Camera Space (Matches Pyglet's 'mv' transform)
        # get_world_to_view_transform() gives the rotation the shader sees
        w2v_mat = self.cameras.get_world_to_view_transform().get_matrix()[0, :3, :3]
        normals_cam = torch.matmul(normals, w2v_mat)

        # 3. Orient toward camera (Eliminates the "Static" noise from PCA)
        # In PyTorch3D Cam Space, Z+ is INTO the screen.
        # If normal points away (Z > 0), flip it.
        flip_mask = normals_cam[:, 2] > 0
        normals_cam = torch.where(flip_mask.unsqueeze(1), -normals_cam, normals_cam)

        # 4. Apply your Pyglet Shader Math exactly:
        # float x = n.x * 0.5 + 0.5; y = 1 - (n.y * 0.5 + 0.5); z = 1 - (n.z * 0.5 + 0.5);
        x = normals_cam[:, 0] * 0.5 + 0.5
        y = 1.0 - (normals_cam[:, 1] * 0.5 + 0.5)
        z = 1.0 - (normals_cam[:, 2] * 0.5 + 0.5)

        mapped_normals = torch.stack([x, y, z], dim=-1)
        mapped_normals = torch.clamp(mapped_normals, 0.0, 1.0)
        
        # 5. Composite
        # permute(1, 0) makes it (Channels, Points) for the compositor
        images = self.compositor(indices, weights, mapped_normals.permute(1, 0))
        
        return images.permute(0, 2, 3, 1)