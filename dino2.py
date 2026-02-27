import torch
import torch.nn.functional as F
from torchvision import transforms as tfs

def init_dino(device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    return model.to(device).eval()

@torch.no_grad()
def get_dino_features(device, dino_model, img_rgb):
    # Resize and Normalize
    transform = tfs.Compose([
        tfs.Resize((518, 518), antialias=True),
        tfs.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img_dino = transform(img_rgb)
    
    # Extract Last Layer
    out = dino_model.get_intermediate_layers(img_dino, n=1)
    features = out[0].half() 
    
    # Reshape (518/14 = 37)
    B, N, Dim = features.shape
    h_patch, w_patch = 37, 37
    features = features.reshape(B, h_patch, w_patch, Dim).permute(0, 3, 1, 2)
    
    # Interpolate to Original Resolution
    features = F.interpolate(
        features.float(), 
        size=img_rgb.shape[-2:], 
        mode='bilinear', 
        align_corners=False
    )
    
    # Normalize
    features = F.normalize(features, dim=1, p=2, eps=1e-6)
    
    return features.half()