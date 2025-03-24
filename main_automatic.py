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
from sklearn.cluster import DBSCAN
from utils import (
    generate_binary_mask,
    delete_irrelevant_detected_pixels,
    fill_little_spaces,
    soften_contours,
    blur_mask,
    delete_files
)

# Cargando variables de entorno
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

# **Función que se ejecuta al cargar una imagen**


def on_image_load(image_path):
    try:
        caption = captioning_model.generate_caption(
            image_path)  # Generar el caption usando el path
        return caption  # Retornar el caption para que se muestre en el campo de texto
    except Exception as e:
        print(f"Error en la generación del caption: {e}")
        return "Error en la generación del caption"


# Construcción de la interfaz en Gradio
with gr.Blocks() as demo:
    with gr.Row():
        img = gr.Image(label="Input Image", type="filepath")
        processed_img = gr.Image(label="Processed Mask", type="filepath")

    with gr.Row():
        generate_button = gr.Button("Generate Mask")

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

    # **Asignar la función para que se ejecute cuando la imagen se cargue**
    img.change(on_image_load, inputs=[img], outputs=text_input)

    # **Manejo de selección de la imagen para generar la máscara**
    def generate_mask(image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(
                    "No se pudo cargar la imagen. Verifica la ruta.")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            #################

            # Convertir a HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

            # Crear máscara de daño por color anómalo
            mask_pink = ((hue > 140) & (hue < 170)) & (sat > 80) & (val > 150)
            mask_yellow = ((hue > 20) & (hue < 40)) & (sat > 80) & (val > 150)
            damage_mask = (mask_pink | mask_yellow).astype(np.uint8) * 255

            # Morfología MUY agresiva
            kernel = np.ones((15, 15), np.uint8)
            damage_mask = cv2.morphologyEx(
                damage_mask, cv2.MORPH_CLOSE, kernel)

            # Componentes conectados
            num_labels, labels = cv2.connectedComponents(damage_mask)

            # Extraer centroides de manchas realmente grandes
            centroids = []
            for label in range(1, num_labels):
                coords = np.column_stack(np.where(labels == label))
                if len(coords) > 150:  # mínimo 150 píxeles por componente
                    y_mean = int(np.mean(coords[:, 0]))
                    x_mean = int(np.mean(coords[:, 1]))
                    centroids.append([x_mean, y_mean])

            # Agrupamiento extremo con DBSCAN
            def cluster_centroids(centroids, eps=100, min_samples=1):
                if len(centroids) == 0:
                    return []
                coords = np.array(centroids)
                clustering = DBSCAN(
                    eps=eps, min_samples=min_samples).fit(coords)
                labels = clustering.labels_

                compacted = []
                for label in np.unique(labels):
                    cluster_coords = coords[labels == label]
                    mean_point = np.mean(
                        cluster_coords, axis=0).astype(int).tolist()
                    compacted.append(mean_point)
                return compacted

            compacted_points = cluster_centroids(centroids, eps=100)

            #################

            binary_mask = None
            for coords in compacted_points:
                print('COORD X: ' + coords[0] + " COORD Y: " + coords[1])
                mask_image = segmentation_model.get_mask_by_pixel(
                    x=coords[0], y=coords[1], image=image)
                mask = generate_binary_mask(mask_image)

                if binary_mask is not None:
                    binary_mask = np.maximum(binary_mask[:, :, 0], mask)

            # Generación de la máscara
            refined_binary_mask = delete_irrelevant_detected_pixels(
                binary_mask)
            without_irrelevant_pixels_mask = fill_little_spaces(
                refined_binary_mask)
            dilated_mask = soften_contours(without_irrelevant_pixels_mask)
            blurred_mask = dilated_mask

            # Guardar máscara procesada
            processed_mask = Image.fromarray(blurred_mask, mode='L')
            processed_mask.save(RUTA_MASCARA)

            return RUTA_MASCARA
        except Exception as e:
            print(f"Error: {e}")
            return None

    # **Reiniciar la máscara al cambiar de imagen**
    def reset_mask(image_path):
        delete_files([RUTA_MASCARA, RUTA_IMAGEN_FINAL])
        return None

    # **Procesar la imagen con la máscara y el texto de entrada**
    def process_final_image(original_image_path, mask_path, text, strength, guidance, negative_prompt):
        try:
            new_image = impainting_model.impaint(
                image_path=original_image_path, mask_path=mask_path, prompt=text, strength=strength, guidance=guidance, negative_prompt=negative_prompt)
            new_image.save(RUTA_IMAGEN_FINAL)
            return RUTA_IMAGEN_FINAL
        except Exception as e:
            print(f"Error: {e}")
            return None

    # **Asignar eventos a la interfaz**
    img.change(reset_mask, inputs=[img], outputs=None)
    generate_button.click(generate_mask, inputs=[img], outputs=processed_img)
    send_button.click(process_final_image, inputs=[
                      img, processed_img, text_input, strength, guidance, negative_prompt], outputs=final_image)

# **Limpiar archivos previos antes de lanzar la aplicación**
delete_files([RUTA_MASCARA, RUTA_IMAGEN_FINAL])

# **Lanzar la interfaz**
demo.launch(debug=True)
