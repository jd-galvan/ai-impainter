from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch


class SDImpainting:
    def __init__(self, device: str):
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to(device)

        self.generator = torch.Generator(device=device).manual_seed(0)

    def impaint(self, image_path: str, mask_path: str, text: str):
        image = load_image(image_path).resize((1024, 1024))
        mask_image = load_image(mask_path).resize((1024, 1024))

        prompt = text

        image = self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            guidance_scale=8.0,
            num_inference_steps=20,  # steps between 15 and 30 work well for us
            strength=0.99,  # make sure to use `strength` below 1.0
            generator=self.generator,
        ).images[0]

        return image
