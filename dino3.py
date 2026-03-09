import torch
import torch.nn.functional as F
from torchvision import transforms as tfs

def init_dino(device):
    # Load the DINOv3 ConvNeXt-Tiny backbone directly
    model = torch.hub.load("facebookresearch/dinov3",
                        #    'dinov3_vits16',weights="weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
                           'dinov3_vitb16',weights="weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    model = model.to(device).eval()
    
    return model

@torch.no_grad()
def get_dino_features_and_score(device, model, img_rgb, score=False):
    transform = tfs.Compose([
        tfs.Resize((512, 512), antialias=True),
        tfs.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img_dino = transform(img_rgb)
    
    # 1. Direct ViT Extraction (Mirrors DINOv2 exactly)
    # This automatically grabs the final layer's normalized patch tokens, skipping the CLS token.
    features = model.get_intermediate_layers(img_dino, n=1)[0]
    
    # 2. Reshape and Interpolate
    B, N, Dim = features.shape
    
    # EXACT MATH: 512 resolution / 16 patch size = 32 patches
    # (If you upgrade to vitb14, change this to 518/14 = 37 like your v2 code)
    h_patch, w_patch = 32, 32 
    
    features = features.reshape(B, h_patch, w_patch, Dim).permute(0, 3, 1, 2)
    
    features = F.interpolate(
        features.float(), 
        size=img_rgb.shape[-2:], 
        mode='bilinear', 
        align_corners=False
    )
    
    # 3. Safe Normalize
    features = F.normalize(features, dim=1, p=2, eps=1e-6)
    
    view_score = 1.0
    class_idx = 0
    
    return features.half(), view_score, class_idx