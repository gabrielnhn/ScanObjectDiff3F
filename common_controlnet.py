import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline, # <--- Back to Text-to-Image
    UniPCMultistepScheduler,
    AutoencoderKL
)

def run_diffusion(text_prompt, depth_image, conditioning_scale=0.75):
    # 1. Load the upgraded VAE (Crucial for good colors/textures from scratch)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    
    # 2. Load the Depth ControlNet
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth",
        torch_dtype=torch.float16
    )
    
    # 3. Load the pure Text2Img Pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        vae=vae, # Inject the good VAE
        torch_dtype=torch.float16
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(0)
    
    # 4. Generate from pure noise + depth map
    image = pipe(
        prompt=text_prompt,
        image=depth_image,           
        controlnet_conditioning_scale=conditioning_scale, # Default 1.0 is too strict for partial shapes
        num_inference_steps=30,
        generator=generator,
        negative_prompt="background, watermark, lowres"
    ).images[0]

    del pipe
    torch.cuda.empty_cache()

    return image


if __name__ == "__main__":
    import torch
    import os
    from huggingface_hub import HfApi
    from pathlib import Path
    from diffusers.utils import load_image
    from PIL import Image
    import numpy as np
    from transformers import pipeline


    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    checkpoint = "lllyasviel/control_v11f1p_sd15_depth"

    image = load_image(
        "https://huggingface.co/lllyasviel/control_v11p_sd15_depth/resolve/main/images/input.png"
    )

    prompt = "Stormtrooper's lecture in beautiful lecture hall"

    depth_estimator = pipeline('depth-estimation')
    image = depth_estimator(image)['depth']
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    control_image = Image.fromarray(image)

    control_image.save("./control.png")

    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(0)
    image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]

    image.save('images/image_out.png')
