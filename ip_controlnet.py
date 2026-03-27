import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    DDIMScheduler, AutoencoderKL, ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline
)
from PIL import Image

# This will now natively read from the folder you dragged and dropped!
from ip_adapter import IPAdapter 

base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "ipmodels/image_encoder/"
ip_ckpt = "ipmodels/ip-adapter_sd15.bin"
device = "cuda"

def init_diffusion():
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth",
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )
    

    
    ip_model = IPAdapter(sd_pipe=pipe,
                         image_encoder_path=image_encoder_path,
                         ip_ckpt=ip_ckpt,
                         device=device)

    # ip_model.pipe.enable_model_cpu_offload()
    # ip_model.pipe.enable_attention_slicing()    
        
    # return pipe
    return ip_model


def run_diffusion(ip_model,
                  best_reference_image,
                  depth_map,
                  current_pov_image,
                  condition_scale,
                  ip_prompt_scale,
                  text_prompt,
                  strength):
    
    image = ip_model.generate(
        pil_image=best_reference_image,
        image=current_pov_image,
        control_image=depth_map, 
        prompt=text_prompt,
        negative_prompt="background, lowres, details",
        scale=ip_prompt_scale, 
        controlnet_conditioning_scale=condition_scale,
        # Img2Img Strength: How much noise to add to current_pov_image. 
        # 0.0 = exact copy of input, 1.0 = completely new image. 
        # For shape completion, 0.7 to 0.9 usually works best.
        strength=strength,
        num_samples=1,
        num_inference_steps=50,
        seed=42,
    )[0]
    
    # del ip_model
    torch.cuda.empty_cache()
    
    return image