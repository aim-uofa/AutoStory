import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import *
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionAdapterPipeline, Adapter


mask = Image.open("motor.png")
prompt = [
    "A black Honda motorcycle parked in front of a garage",
    "A red-blue Honda motorcycle parked in front of a garage",
    "A green Honda motorcycle parked in a desert",
]

# model_name = "CompVis/stable-diffusion-v1-4"
model_name = "RzZ/sd-v1-4-adapter"
pipe = StableDiffusionAdapterPipeline.from_pretrained(model_name, torch_dtype=torch.float16)


pipe.to("cuda")

images = pipe(prompt, [mask] * len(prompt)).images

plt.subplot(2, 2, 1)
plt.imshow(mask)

for i, image in enumerate(images):
    plt.subplot(2, 2, 2 + i)
    plt.imshow(image)
    plt.title(prompt[i], fontsize=24)
plt.show()