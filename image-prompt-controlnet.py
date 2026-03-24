from types import MethodType
import torch
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel
from PIL import Image
from ip_adapter import IPAdapter



base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "ipmodels/image_encoder/"
ip_ckpt = "ipmodels/ip-adapter_sd15.bin"
device = "cuda"

def init_diffusion():
    # https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter_controlnet_demo_new.ipynb
    
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
    
    # load controlnet
    controlnet_model_path = "lllyasviel/control_v11f1p_sd15_depth"
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_path,
        torch_dtype=torch.float16,
        use_safetensors=True
        )
    # load SD pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    
    ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
    return ip_model


def run_diffusion(
    ip_model, 
    input_image,
    depth_map,
):
    image = ip_model.generate(
        pil_image=input_image,
        image=depth_map,
        num_samples=1,
        num_inference_steps=50,
        seed=42
    )
    return image
