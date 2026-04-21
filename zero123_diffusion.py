import torch
import copy
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, ControlNetModel

device = "cuda"

def run_diffusion(cond_image, text_prompt=""):
    print("Initializing Zero123++ v1.2 Base Pipeline...")
    base_pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2", 
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16
    )
    
    # Feel free to tune the scheduler
    base_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        base_pipeline.scheduler.config, 
        timestep_spacing='trailing'
    )

    base_pipeline.to(device)
    
    print("Generating RGB Multi-Views (Shape Completion)...")
    genimg = base_pipeline(
        cond_image,
        prompt=text_prompt,
        negative_prompt="background, lowres, details, watermark",
        guidance_scale=4.0,
        num_inference_steps=75,
        width=640,
        height=960
    ).images[0]
    
    
    print("Initializing Normal ControlNet Pipeline...")
    normal_pipeline = copy.copy(base_pipeline)
    normal_pipeline.add_controlnet(
        ControlNetModel.from_pretrained(
            "sudo-ai/controlnet-zp12-normal-gen-v1", 
            torch_dtype=torch.float16
        ), 
        conditioning_scale=1.0
    )
    
    del base_pipeline
    torch.cuda.empty_cache()
    
    normal_pipeline.to(device)
    print("Generating View-Space Normal Maps...")
    normalimg = normal_pipeline(
        cond_image,
        depth_image=genimg, # The normal pipeline takes the generated RGB grid as its condition
        prompt=text_prompt,
        negative_prompt="background, lowres, details, watermark",
        guidance_scale=4.0, 
        num_inference_steps=75,
        width=640,
        height=960
    ).images[0]
    
    del normal_pipeline
    torch.cuda.empty_cache()
    
    return genimg, normalimg