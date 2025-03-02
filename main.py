import gradio as gr
from PIL import Image
import numpy as np
import cv2
import io
from libs.sam2.model import SAM2
from libs.stable_diffusion.impaint.model import SDImpainting
from utils import (
    generate_binary_mask,
    delete_irrelevant_detected_pixels,
    fill_little_spaces,
    soften_contours,
    blur_mask,
    delete_files
)

# Rutas de archivos generados
RUTA_MASCARA = "processed_mask.png"
RUTA_IMAGEN_FINAL = "final_output.png"

# Configuración del dispositivo para modelos
DEVICE = "cuda:1"

# Cargar modelos
segmentation_model = SAM2(DEVICE)
impainting_model = SDImpainting(DEVICE)

# Construcción de la interfaz en Gradio
with gr.Blocks() as demo:
    with gr.Row():
        img = gr.Image(label="Input Image", type="filepath", interactive=True)
        processed_img = gr.Image(label="Processed Mask", type="filepath")

    with gr.Row():
        text_input = gr.Textbox(label="Enter prompt",
                                placeholder="Write something here...")

    with gr.Row():
        send_button = gr.Button("Generate Final Image")

    with gr.Row():
        final_image = gr.Image(label="Final Output", type="filepath")

    # Manejo de selección de la imagen para generar la máscara
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
            refined_binary_mask = delete_irrelevant_detected_pixels(
                binary_mask)
            without_irrelevant_pixels_mask = fill_little_spaces(
                refined_binary_mask)
            dilated_mask = soften_contours(without_irrelevant_pixels_mask)
            blurred_mask = blur_mask(dilated_mask)

            # Mezcla de máscaras previas si existen
            old_mask = cv2.imread(RUTA_MASCARA)
            if old_mask is not None:
                blurred_mask = np.maximum(old_mask[:, :, 0], blurred_mask)

            # Guardar máscara procesada
            processed_mask = Image.fromarray(blurred_mask, mode='L')
            processed_mask.save(RUTA_MASCARA)

            return RUTA_MASCARA
        except Exception as e:
            print(f"Error: {e}")
            return None

    # Reiniciar la máscara al cambiar de imagen
    def reset_mask(image_path):
        delete_files([RUTA_MASCARA, RUTA_IMAGEN_FINAL])
        return None

    # Procesar la imagen con la máscara y el texto de entrada
    def process_final_image(original_image_path, mask_path, text):
        try:
            new_image = impainting_model.impaint(
                image_path=original_image_path, mask_path=mask_path, text=text)
            new_image.save(RUTA_IMAGEN_FINAL)
            return RUTA_IMAGEN_FINAL
        except Exception as e:
            print(f"Error: {e}")
            return None

    # Asignar eventos a la interfaz
    img.select(on_select, inputs=[img], outputs=processed_img)
    img.change(reset_mask, inputs=[img], outputs=None)
    send_button.click(process_final_image, inputs=[
                      img, processed_img, text_input], outputs=final_image)

# Limpiar archivos previos antes de lanzar la aplicación
delete_files([RUTA_MASCARA, RUTA_IMAGEN_FINAL])

# Lanzar la interfaz
demo.launch(debug=True)
