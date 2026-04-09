from transformers import CLIPProcessor, CLIPModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_name = "openai/clip-vit-base-patch32"


def init_clip():
    processor = CLIPProcessor.from_pretrained(clip_name)
    print("Loading CLIP to GPU...")
    model = CLIPModel.from_pretrained(clip_name).to(device)
    model.eval()
    return model, processor

@torch.inference_mode()
def clip_score(
    clip_model,
    processor, 
    image,
    prompt="A clear, comprehensive, and high-quality view showing the full shape of the object"
):
    inputs = processor(
        text=[prompt], 
        images=image, 
        return_tensors="pt", 
        padding=True
    ).to(device)

    outputs = clip_model(**inputs)
    
    score = outputs.logits_per_image[0][0].item() 
    return score