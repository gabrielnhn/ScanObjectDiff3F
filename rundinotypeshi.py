import torch
import numpy as np
from torchvision import transforms as tfs
from PIL import Image
from sklearn.decomposition import PCA

# 1. Setup
patch_size = 14
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_dino():
    # Load DINOv2 - ViT-B/14
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model = model.to(device).eval()
    return model

# 2. Image Prep
img_path = "./cats.png"
# Create a dummy image if file doesn't exist for testing
# Image.new('RGB', (500, 500), color='red').save(img_path) 

img = Image.open(img_path).convert("RGB")
w, h = img.size

# Resize to multiple of patch_size for clean reshaping
# 518 is standard for DINOv2 (37 * 14 = 518)
transform = tfs.Compose([
    # tfs.Resize((518, 518)),
    tfs.Resize((50*14, 50*14)),
    tfs.ToTensor(),
    tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

img_tensor = transform(img).unsqueeze(0).to(device) # Add batch dim: (1, 3, 518, 518)

# 3. Inference
dino = init_dino()

with torch.no_grad():
    # forward_features gives us the patch tokens (spatial features)
    # Dictionary keys: 'x_norm_clstoken', 'x_norm_patchtokens'
    features_dict = dino.forward_features(img_tensor)
    features = features_dict['x_norm_patchtokens'] # Shape: (1, 1369, 768)

# 4. Process Features (PCA)
# Flatten to (N_patches, Dim) -> (1369, 768)
features = features.squeeze(0).cpu().numpy()

# Fit PCA to get the first 3 principal components (to map to RGB)
pca = PCA(n_components=3)
pca_features = pca.fit_transform(features)

# Min-Max normalize to 0-255 for image saving
pca_features = (pca_features - pca_features.min(0)) / (pca_features.max(0) - pca_features.min(0))
pca_features = (pca_features * 255).astype(np.uint8)

# 5. Reshape and Save
# 518 / 14 = 37 patches
grid_size = 50*14 // patch_size 
pca_img = pca_features.reshape(grid_size, grid_size, 3)

# Resize back to original image size for better viewing
result_img = Image.fromarray(pca_img)
result_img = result_img.resize((w, h), resample=Image.NEAREST)

result_img.save("cats_result_pca.png")
print("Saved visualization to cats_result_pca.png")