import torch
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, ControlNetModel

device = "cuda"

def init_diffusion(conditioning_scale=0.75):
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.1",
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16
    )
    pipeline.add_controlnet(
        ControlNetModel.from_pretrained(
            "sudo-ai/controlnet-zp11-depth-v1",
            torch_dtype=torch.float16
        ),
        conditioning_scale=0.75
    )
    
    # Feel free to tune the scheduler
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config,
        timestep_spacing='trailing'
    )

    # idk if I should move it to cuda or not
    # should one enable attention slicing here or something?
    pipeline.to(device)
    
    return pipeline


def run_diffusion(pipe,
                  best_reference_image,
                  depth_map,
                  text_prompt,
                #   strength
                  ):
    
    # reference pipeline call
    # def __call__(
    #     image: Image.Image = None,
    #     prompt = "",
    #     num_images_per_prompt: Optional[int] = 1,
    #     guidance_scale=4.0,
    #     depth_image: Image.Image = None,
    #     output_type: Optional[str] = "pil",
    # ):
    
    image = pipe(
        best_reference_image,
        depth_image=depth_map,
        prompt=text_prompt,
        # hopefully kwargs work
        # strength=strength,
        # seed=42,
        negative_prompt="background, lowres, details, watermark",        
    ).images[0]
    
    # torch.cuda.empty_cache()
    
    return image