import io
import cv2
import os
import gradio as gr
from dotenv import load_dotenv
from PIL import Image
import numpy as np
from libs.sam2.model import SAM2
from libs.stable_diffusion.impaint.model import SDImpainting
from libs.blip.model import BLIP
from utils import (
    generate_binary_mask,
    delete_irrelevant_detected_pixels,
    fill_little_spaces,
    soften_contours,
    blur_mask,
    delete_files
)

# Cargar variables de entorno
load_dotenv()

# Rutas de archivos generados
RUTA_MASCARA = "processed_mask.png"
RUTA_IMAGEN_FINAL = "final_output.png"

# Configuración del dispositivo para modelos
DEVICE = os.environ.get("CUDA_DEVICE")
print(f"DEVICE {DEVICE}")

# Cargar modelos
captioning_model = BLIP(DEVICE)
segmentation_model = SAM2(DEVICE)
impainting_model = SDImpainting(DEVICE)

# Función que se ejecuta al cargar una imagen


def on_image_load(image_path):
    try:
        caption = captioning_model.generate_caption(image_path)
        return caption
    except Exception as e:
        print(f"Error en la generación del caption: {e}")
        return "Error en la generación del caption"

# Función para remover píxeles irrelevantes


def remove_irrelevant_pixels_func(image_path):
    try:
        # Se asume que la imagen es una máscara en escala de grises
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("No se pudo cargar la imagen.")
        processed = delete_irrelevant_detected_pixels(image)
        output_path = "processed_removed.png"
        cv2.imwrite(output_path, processed)
        return output_path
    except Exception as e:
        print(f"Error en remove_irrelevant_pixels_func: {e}")
        return None

# Función para aplicar fill_little_spaces al reenviar de la 3era a la 4ta imagen


def apply_fill_little_spaces(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("No se pudo cargar la imagen.")
        processed = fill_little_spaces(image)
        output_path = "filled_spaces.png"
        cv2.imwrite(output_path, processed)
        return output_path
    except Exception as e:
        print(f"Error en apply_fill_little_spaces: {e}")
        return None

# Función para aplicar soften_contours al reenviar de la 4ta a la 5ta imagen


def apply_soften_contours(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("No se pudo cargar la imagen.")
        processed = soften_contours(image)
        output_path = "softened_contours.png"
        cv2.imwrite(output_path, processed)
        return output_path
    except Exception as e:
        print(f"Error en apply_soften_contours: {e}")
        return None

# Procesar la imagen final usando la imagen original y la quinta imagen


def process_final_image(original_image_path, processed_image_path, text, strength, guidance, negative_prompt):
    try:
        new_image = impainting_model.impaint(
            image_path=original_image_path,
            mask_path=processed_image_path,
            prompt=text,
            strength=strength,
            guidance=guidance,
            negative_prompt=negative_prompt
        )
        new_image.save(RUTA_IMAGEN_FINAL)
        return RUTA_IMAGEN_FINAL
    except Exception as e:
        print(f"Error: {e}")
        return None


# Construcción de la interfaz en Gradio
with gr.Blocks() as demo:
    # Fila 1: Imagen de entrada y máscara procesada
    with gr.Row():
        img = gr.Image(label="Input Image", type="filepath", interactive=True)
        processed_img = gr.Image(label="Processed Mask", type="filepath")

    # Botón para remover píxeles irrelevantes (entre la fila 1 y la fila 2)
    with gr.Row():
        remove_button = gr.Button("Remove irrelevant pixels")

    # Fila 2: Contendrá la tercera y cuarta imagen, cada una en una columna con su respectivo botón debajo
    with gr.Row():
        with gr.Column():
            third_img = gr.Image(label="Third Image", type="filepath")
            to_fourth_btn = gr.Button("Fill little spaces")
        with gr.Column():
            fourth_img = gr.Image(label="Fourth Image", type="filepath")
            to_fifth_btn = gr.Button("Soften contours")

    # Fila 3: Contendrá la quinta imagen
    with gr.Row():
        fifth_img = gr.Image(label="Fifth Image", type="filepath")

    # Elementos adicionales para la generación de la imagen final
    with gr.Row():
        text_input = gr.Textbox(label="Enter prompt",
                                placeholder="Write prompt for impainting...")

    with gr.Row():
        strength = gr.Slider(minimum=0.0, maximum=1.0,
                             value=0.99, label="Strength", interactive=True)
        guidance = gr.Slider(minimum=0.0, maximum=50.0,
                             value=7.0, label="Guidance Scale", interactive=True)

    with gr.Row():
        negative_prompt = gr.Textbox(
            label="Negative prompt", placeholder="Write negative prompt...")

    with gr.Row():
        send_button = gr.Button("Generate Final Image")

    with gr.Row():
        final_image = gr.Image(label="Final Output", type="filepath")

    # Al cargar la imagen se genera el caption en el textbox
    img.change(on_image_load, inputs=[img], outputs=text_input)

    # Función para seleccionar la imagen y generar la máscara a partir de un clic
    def on_select(image_path, evt: gr.SelectData):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(
                    "No se pudo cargar la imagen. Verifica la ruta.")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            coords = evt.index  # Coordenadas del píxel seleccionado

            # Generación de la máscara
            mask_image = segmentation_model.get_mask_by_pixel(
                x=coords[0], y=coords[1], image=image)
            binary_mask = generate_binary_mask(mask_image)

            # Mezcla con máscara previa si existe
            old_mask = cv2.imread(RUTA_MASCARA)
            if old_mask is not None:
                binary_mask = np.maximum(old_mask[:, :, 0], binary_mask)

            # Guardar máscara procesada
            processed_mask = Image.fromarray(binary_mask, mode='L')
            processed_mask.save(RUTA_MASCARA)
            return RUTA_MASCARA
        except Exception as e:
            print(f"Error: {e}")
            return None

    # Reiniciar la máscara al cambiar de imagen
    def reset_mask(image_path):
        delete_files([RUTA_MASCARA, RUTA_IMAGEN_FINAL])
        return None

    # Asignar eventos a la interfaz
    img.select(on_select, inputs=[img], outputs=processed_img)
    img.change(reset_mask, inputs=[img], outputs=None)

    # Botón para remover píxeles irrelevantes: toma la imagen del Processed Mask y la envía al widget Third Image
    remove_button.click(remove_irrelevant_pixels_func, inputs=[
                        processed_img], outputs=third_img)

    # Botones para reenviar la imagen de un widget al siguiente:
    # Se aplica fill_little_spaces al enviar de la 3era a la 4ta imagen
    to_fourth_btn.click(apply_fill_little_spaces, inputs=[
                        third_img], outputs=fourth_img)
    # Se aplica soften_contours al enviar de la 4ta a la 5ta imagen
    to_fifth_btn.click(apply_soften_contours, inputs=[
                       fourth_img], outputs=fifth_img)

    # Botón para generar la imagen final usando la imagen original y la quinta imagen
    send_button.click(process_final_image, inputs=[
                      img, fifth_img, text_input, strength, guidance, negative_prompt], outputs=final_image)

# Limpiar archivos previos antes de lanzar la aplicación
delete_files([RUTA_MASCARA, RUTA_IMAGEN_FINAL])

# Lanzar la interfaz
demo.launch(debug=True)
