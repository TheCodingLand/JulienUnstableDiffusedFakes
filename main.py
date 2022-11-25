from typing import List
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from pydantic import BaseModel
model_id = "CompVis/stable-diffusion-v1-5"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

#prompt = "a photo of an astronaut riding a horse on mars"

prompt = "a beautiful girl in anime style looking at a sunset in a realistic style"
prompt = "A 4 steps storyboard of a girl doing a kung-fu fight against 3 men"
with autocast("cuda"):
    images : List["Image"]= pipe(prompt, guidance_scale=7.5).images

print(type(images[0]))
image[0]
fname = prompt.replace(' ', '_')
for i, image in enumerate(images[:4]):
    image.save(f"{fname}_{i}.png")

class CustomStableDiffusionModelTrainingPipeline(BaseModel):
    source_model_id :str
    training_data_path :str
    def finetune_stablediffusion(self) -> None:
        ...
    def load_data(self):
        ...
