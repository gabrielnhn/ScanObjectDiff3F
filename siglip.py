from transformers import AutoProcessor, AutoModel
import torch
device = "cuda"
siglip_name = "google/siglip-base-patch16-224"
sl_processor = AutoProcessor.from_pretrained(siglip_name)

def init_siglip():
    sl_model = AutoModel.from_pretrained(siglip_name).to(device)
    sl_model.enable_model_cpu_offload()
    sl_model.enable_attention_slicing()
    sl_model.eval()
    return sl_model

@torch.inference_mode()
def siglip_score(sl_model, image,
    prompt="A clear, comprehensive, and high-quality view showing the full shape of the object"
):

    # The text we want the image to match perfectly
    prompt = [prompt]
    inputs = sl_processor(
        text=prompt, 
        images=image, 
        padding="max_length", 
        return_tensors="pt"
    ).to(device)

    # Get Score
    outputs = sl_model(**inputs)
    # SigLIP uses Sigmoid for independent probabilities
    probs = torch.sigmoid(outputs.logits_per_image) 
    score = probs[0][0].item()
    return score