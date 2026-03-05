from diffusers import DiffusionPipeline
import torch

# pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
# pipeline = DiffusionPipeline.from_pretrained("/nas2/wwen/weights/stable-diffusion/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline = DiffusionPipeline.from_pretrained("/nas2/wwen/weights/t2i-adaptor/anything-v4.0", torch_dtype=torch.float32)
pipeline.to("cuda")
# image = pipeline("An image of a squirrel in Picasso style").images[0]
# image = pipeline("a teddy bear sitting next to a bird").images[0]

prompt = "a beautiful little bird"  # with bright feathers of red, blue, and yellow, and sparkling eyes of mischief"

for idx in range(10):
    # image = pipeline("a teddy bear sitting next to a bird").images[0]
    image = pipeline(prompt).images[0]
    image.save(f"tmp1_bird_anything_{idx}.png")
