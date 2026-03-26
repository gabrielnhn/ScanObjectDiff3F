# thank you Gemini for this module :)

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Base-hf")
device = "cuda" if torch.cuda.is_available() else "cpu"

def init_depther():
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Base-hf").to(device)
    model.eval()
    return model

@torch.inference_mode()
def get_depth_anything_map(model, image_rgb_pil):
    inputs = processor(images=image_rgb_pil, return_tensors="pt").to(device)

    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

    # 3. Interpolate back to your EXACT original image resolution
    # PIL sizes are (width, height), PyTorch needs (height, width)
    depth_resized = F.interpolate(
        predicted_depth.unsqueeze(1),
        size=image_rgb_pil.size[::-1], 
        mode="bicubic",
        align_corners=False,
    )
    
    # 4. Squeeze out the batch/channel dimensions and convert to a CPU Numpy array
    depth_numpy = depth_resized.squeeze().cpu().numpy()
    
    return depth_numpy