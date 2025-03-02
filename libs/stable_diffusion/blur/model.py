# import torch
# from diffusers import AutoPipelineForInpainting
# from diffusers.utils import load_image
# from PIL import Image


# class SDBlur:
#     def __init__(self, device: str):
#         self.pipeline = AutoPipelineForInpainting.from_pretrained(
#             "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)

#     def blur(self, mask_path: str, blur_factor: int = 33):
#         mask = load_image(mask_path)
#         blurred_mask = self.pipeline.mask_processor.blur(
#             mask, blur_factor=blur_factor)
#         return blurred_mask
