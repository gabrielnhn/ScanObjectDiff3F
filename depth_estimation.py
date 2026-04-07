# thank you Gemini for this module :)

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

model_string = "Intel/dpt-hybrid-midas" 

processor = AutoImageProcessor.from_pretrained(model_string)
device = "cuda" if torch.cuda.is_available() else "cpu"

def init_depther():
    print("Loading MiDaS DPT-Large to GPU...")
    model = AutoModelForDepthEstimation.from_pretrained(model_string).to(device)
    model.eval()
    return model

@torch.inference_mode()
def get_depth_map(model, image_rgb_pil):
    inputs = processor(images=image_rgb_pil, return_tensors="pt").to(device)

    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

    depth_resized = F.interpolate(
        predicted_depth.unsqueeze(1),
        size=image_rgb_pil.size[::-1], 
        mode="bicubic",
        align_corners=False,
    )
    
    depth_cpu = depth_resized.squeeze().cpu()
    
    return depth_cpu