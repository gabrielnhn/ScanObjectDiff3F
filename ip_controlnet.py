import torch
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel
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
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
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


def run_diffusion(ip_model, input_image, depth_map,
                  condition_scale,
                  ip_prompt_scale):
    print("Loading IP-Adapter Image Encoder to GPU...")
    # Initialize Tencent's IPAdapter locally
    
    print("Generating Image...")
    # Generate
    image = ip_model.generate(
        pil_image=input_image,
        image=depth_map,
        prompt="the back of sofa, back of a couch, png, white background",
        negative_prompt="background, lowres, details, facing viewer",
        scale=ip_prompt_scale, # how strongly IP-Adapter affects image
        num_samples=1,
        num_inference_steps=50,
        seed=42,
        controlnet_conditioning_scale=condition_scale,
    )[0]
    
    # del ip_model
    torch.cuda.empty_cache()
    
    return image