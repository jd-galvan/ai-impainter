import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
import os
from PIL import Image
import numpy as np

MODEL_IMG_DIM = 1024

class FluxImpainting:
    def __init__(self, device: str):
        self.pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16, token=os.environ.get("HUGGINGFACE_HUB_TOKEN")).to(device)
        print("Modelo Flux Impainting descargado ✅")

    def impaint(self, image_path: str, mask_path: str, prompt: str, negative_prompt: str, strength: float, guidance: float, steps: int):
        # Carga las imágenes originales
        original_image = load_image(image_path)
        original_mask = load_image(mask_path)

        # Guarda las dimensiones originales
        orig_width, orig_height = original_image.size

        # Calcula el factor de escala para ajustar la imagen al canvas de 1024x1024 sin distorsionar la relación de aspecto
        scale = min(MODEL_IMG_DIM / orig_width, MODEL_IMG_DIM / orig_height)
        new_size = (int(orig_width * scale), int(orig_height * scale))

        # Redimensiona la imagen y la máscara manteniendo la relación de aspecto
        resized_image = original_image.resize(new_size, Image.LANCZOS)
        resized_mask = original_mask.resize(new_size, Image.LANCZOS)

        # Crea un canvas de 1024x1024 para la imagen y la máscara
        canvas_image = Image.new("RGB", (MODEL_IMG_DIM, MODEL_IMG_DIM), (255, 255, 255))
        # Ajusta el color de fondo de la máscara según necesites
        canvas_mask = Image.new("RGB", (MODEL_IMG_DIM, MODEL_IMG_DIM), (0, 0, 0))

        # Calcula las posiciones para centrar la imagen redimensionada en el canvas
        paste_x = (MODEL_IMG_DIM - new_size[0]) // 2
        paste_y = (MODEL_IMG_DIM - new_size[1]) // 2

        canvas_image.paste(resized_image, (paste_x, paste_y))
        canvas_mask.paste(resized_mask, (paste_x, paste_y))

        # Ejecuta el pipeline de inpainting en el canvas
        result_canvas = self.pipe(
            prompt=prompt,
            image=canvas_image,
            mask_image=canvas_mask,
            height=MODEL_IMG_DIM,
            width=MODEL_IMG_DIM,
            guidance_scale=guidance,
            num_inference_steps=steps,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]

        # Recorta la parte central que contiene la imagen original redimensionada
        result_crop = result_canvas.crop(
            (paste_x, paste_y, paste_x + new_size[0], paste_y + new_size[1]))
        # Reescala la imagen resultante a las dimensiones originales
        result_crop = result_crop.resize(
            (orig_width, orig_height), Image.LANCZOS)

        # Fusiona usando la máscara binaria para evitar alteraciones, sobre todo en rostros, por el impainting
        original_np = np.array(original_image)
        result_np = np.array(result_crop)
        mask_np = np.array(original_mask)

        white_mask = mask_np == 255  # Solo cambia donde la máscara es blanca

        final_np = original_np.copy()
        final_np[white_mask] = result_np[white_mask]

        final_result = Image.fromarray(final_np)

        return final_result
