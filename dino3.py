import torch
import torch.nn.functional as F
from torchvision import transforms as tfs

def init_dino(device):
    # Load the DINOv3 ConvNeXt-Tiny backbone directly
    model = torch.hub.load("facebookresearch/dinov3", 'dinov3_convnext_tiny')
    model = model.to(device).eval()
    
    return model

@torch.no_grad()
def get_dino_features_and_score(device, model, img_rgb, score=False):
    transform = tfs.Compose([
        tfs.Resize((512, 512), antialias=True),
        tfs.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img_dino = transform(img_rgb)
    
    # 1. Forward pass BYPASSING Global Average Pooling
    if hasattr(model, 'forward_features'):
        # For ConvNeXts, this returns the dense (B, C, H, W) spatial map
        features = model.forward_features(img_dino)
    elif hasattr(model, 'get_intermediate_layers'):
        # Fallback for ViT architectures
        features = model.get_intermediate_layers(img_dino, n=1)[0]
    else:
        # Extreme fallback
        features = model(img_dino)
    
    # Extract from dict if Meta wrapped it
    if isinstance(features, dict):
        features = features.get("x_norm_patchtokens", list(features.values())[-1])
        
    # 2. Check Dimensions to ensure we have spatial data
    if features.ndim == 2:
        raise RuntimeError("Features are 2D! The model applied Global Average Pooling. We need the raw spatial map.")
    elif features.ndim == 3: 
        # Flattened sequence (B, N, C) -> Spatial Map (B, C, H, W)
        B, N, C = features.shape
        grid_size = int(N ** 0.5)
        features = features.reshape(B, grid_size, grid_size, C).permute(0, 3, 1, 2)
    # If ndim == 4, it's already (B, C, H, W) from ConvNeXt and ready to go!
    
    # 3. Interpolate back to your exact render resolution
    features = F.interpolate(
        features.float(), 
        size=img_rgb.shape[-2:], 
        mode='bilinear', 
        align_corners=False
    )
    
    # 4. Safe normalize
    features = F.normalize(features, dim=1, p=2, eps=1e-6)
    
    view_score = 1.0
    class_idx = 0
    
    return features.half(), view_score, class_idx