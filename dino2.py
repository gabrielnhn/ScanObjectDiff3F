import torch
import torch.nn.functional as F
from torchvision import transforms as tfs

hook_features = {}

def hook_fn(module, input, output):
    # 'output' = final tokens right before classification head
    hook_features['tokens'] = output

def init_dino(device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_lc")
    model = model.to(device).eval()
    
    # attach hook to the backbone final LayerNorm
    model.backbone.norm.register_forward_hook(hook_fn)
    
    return model

@torch.no_grad()
def get_dino_features_and_score(device, model, img_rgb, score=None):
    transform = tfs.Compose([
        tfs.Resize((518, 518), antialias=True),
        tfs.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img_dino = transform(img_rgb)
    
    # forward pass (triggers hook!)
    logits = model(img_dino)
    
    # Semantic View Score via MOST confident guess.
    probs = F.softmax(logits, dim=-1)
    
    max_prob, class_idx = probs.max(dim=-1)
    view_score = max_prob.item()
    
    # view_score = probs.max(dim=-1).values.item()
    
    # retrieve the Intercepted Patch Features
    # Shape is (Batch, Tokens, Dim). Token 0 is the [CLS] token, Tokens 1+ are the patches.
    raw_tokens = hook_features['tokens']
    features = raw_tokens[:, 1:, :].half() # Grab patches, skip CLS
    
    # Reshape and Interpolate
    B, N, Dim = features.shape
    h_patch, w_patch = 37, 37
    features = features.reshape(B, h_patch, w_patch, Dim).permute(0, 3, 1, 2)
    
    features = F.interpolate(
        features.float(), 
        size=img_rgb.shape[-2:], 
        mode='bilinear', 
        align_corners=False
    )
    
    features = F.normalize(features, dim=1, p=2, eps=1e-6)
    
    view_score = 1
    class_idx = 0
    
    return features.half(), view_score, class_idx