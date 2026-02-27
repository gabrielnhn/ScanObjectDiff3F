import torch
import numpy as np
from torchvision import transforms as tfs

patch_size = 14

def init_dino(device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model = model.to(device).eval()
    return model

@torch.no_grad
def get_dino_features(device, dino_model, img, grid=None):
    # img: (1, 3, H, W)
    
    # 1. Resize/Normalize for DINO
    # DINO prefers multiples of 14. 518 is standard.
    transform = tfs.Compose([
        tfs.Resize((518, 518), antialias=True),
        tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img_dino = transform(img)
    
    # 2. Extract Features (Layer n=1 is the last layer)
    # You can try n=4 and taking the 4th-to-last for more geometry info.
    raw_features = dino_model.get_intermediate_layers(img_dino, n=1)[0].half()
    
    # 3. Reshape to Patch Grid
    h_patch = int(img_dino.shape[2] / patch_size)
    w_patch = int(img_dino.shape[3] / patch_size)
    dim = raw_features.shape[-1]
    
    # (B, N_patches, Dim) -> (B, Dim, H_patch, W_patch)
    features = raw_features.reshape(img_dino.shape[0], h_patch, w_patch, dim).permute(0, 3, 1, 2)
    
    # 4. Upsample back to ORIGINAL Image Size (img.shape)
    # This aligns features perfectly with your rendered pixels.
    features = torch.nn.functional.interpolate(
        features.float(), 
        size=(img.shape[2], img.shape[3]), # Target size (e.g., 512, 512)
        mode='bilinear', 
        align_corners=False # Standard for image resizing
    )
    
    # 5. Normalize
    features = torch.nn.functional.normalize(features, dim=1)
    
    return features.half() # Returns (1, Dim, H, W)