# import torch
# import requests
# from PIL import Image
# from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

# # Load the pipeline
# pipeline = DiffusionPipeline.from_pretrained(
#     "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
#     torch_dtype=torch.float16
# )

# print("pipelien loaded")

# # Feel free to tune the scheduler!
# # `timestep_spacing` parameter is not supported in older versions of `diffusers`
# # so there may be performance degradations
# # We recommend using `diffusers==0.20.2`
# pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
#     pipeline.scheduler.config, timestep_spacing='trailing'
# )
# pipeline.to('cuda:0')
# print("scheduler loaded")

# # Download an example image.
# cond = Image.open(requests.get("https://d.skis.ltd/nrp/sample-data/lysol.png", stream=True).raw)

# print("running pielien")
# # Run the pipeline!
# result = pipeline(cond, num_inference_steps=75).images[0]
# # for general real and synthetic images of general objects
# print("ran pielien")
# # usually it is enough to have around 28 inference steps
# # for images with delicate details like faces (real or anime)
# # you may need 75-100 steps for the details to construct

# # result.show()
# result.save("output.png")

import torch
import requests
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, ControlNetModel

# Load the pipeline
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16
)
pipeline.add_controlnet(ControlNetModel.from_pretrained(
    "sudo-ai/controlnet-zp11-depth-v1", torch_dtype=torch.float16
), conditioning_scale=0.75)
# Feel free to tune the scheduler
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)
pipeline.to('cuda')
# Run the pipeline
cond = Image.open(requests.get("https://d.skis.ltd/nrp/sample-data/0_cond.png", stream=True).raw)
depth = Image.open(requests.get("https://d.skis.ltd/nrp/sample-data/0_depth.png", stream=True).raw)
result = pipeline(cond, depth_image=depth).images[0]
# result.show()
result.save("output.png")
