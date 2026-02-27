import torch
import numpy as np
from torchvision import transforms as tfs

patch_size = 14

def init_dino(device):
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitb14",
    )
    model = model.to(device).eval()
    return model

@torch.no_grad
def get_dino_features(device, dino_model, img, grid):
    # img comes in as (1, C, H, W)
    
    # 1. Ensure we only have RGB (Drop Alpha if present)
    if img.shape[1] > 3:
        img = img[:, :3, :, :]
        
    # 2. Define Transforms
    # Note: We resize to 518x518 as preferred by DINOv2
    transform = tfs.Compose([
        tfs.Resize((518, 518), antialias=True),
        tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # 3. Apply Transform
    # img is already a tensor on device, so we just run the transform
    img = transform(img)
    
    # 4. Forward Pass
    # output: (Batch, N_patches, Dim)
    features = dino_model.get_intermediate_layers(img, n=1)[0].half()
    
    # 5. Reshape to Spatial Dimensions
    # DINOv2 (VitB14) on 518x518 yields 37x37 patches (518/14 = 37)
    h, w = int(img.shape[2] / patch_size), int(img.shape[3] / patch_size)
    dim = features.shape[-1]
    
    # (B, H*W, Dim) -> (B, Dim, H, W)
    features = features.reshape(img.shape[0], h, w, dim).permute(0, 3, 1, 2)
    
    # 6. Sample features back to the original pixel grid
    features = torch.nn.functional.grid_sample(
        features.float(), # Grid sample usually needs float32
        grid.float(), 
        align_corners=False
    ).reshape(1, dim, -1)
    
    features = torch.nn.functional.normalize(features, dim=1)
    return features.half()