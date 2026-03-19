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
    
    # THE SECRET SAUCE: Grab the last 4 layers
    layers = model.get_intermediate_layers(img_dino, n=4)
    
    # THE MEMORY SAVER: Average them instead of concatenating!
    # This blends the features but keeps the dimension at exactly 768.
    features = torch.stack(layers).mean(dim=0)
    
    B, N, Dim = features.shape
    h_patch, w_patch = 32, 32 
    
    features = features.reshape(B, h_patch, w_patch, Dim).permute(0, 3, 1, 2)
    
    # Now this is only interpolating 768 channels, saving ~2.5 GB of VRAM
    features = F.interpolate(
        features.float(), 
        size=img_rgb.shape[-2:], 
        mode='bilinear', 
        align_corners=False
    )
    
    features = F.normalize(features, dim=1, p=2, eps=1e-6)
    
    view_score = 1.0
    class_idx = 0
    
    return features.half(), view_score, class_idx