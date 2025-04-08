import io
import cv2
import os
import re
import glob
import gradio as gr
from dotenv import load_dotenv
from PIL import Image
import numpy as np
from libs.sam2.model import SAM2
from libs.stable_diffusion.impaint.model import SDImpainting
from libs.blip.model import BLIP
from libs.yolov8.model import YOLOV8
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
#captioning_model = BLIP(DEVICE)
segmentation_model = SAM2(DEVICE)
impainting_model = SDImpainting(DEVICE)
yolo_model = YOLOV8(device=DEVICE)

# Lista Yolos entrenado
def list_best_pt():
    paths = glob.glob("./tools/trainer/yolov8/runs/detect/full_dataset_yolov8n*/weights/best.pt")

    # Función para extraer el número de la carpeta (train, train2, etc.)
    def extract_number(path):
        match = re.search(r'full_dataset_yolov8n(\d*)', path)
        return int(match.group(1)) if match and match.group(1) else 0

    # Ordenar de forma descendente según el número
    paths.sort(key=extract_number, reverse=True)

    print(f"Modelos detectados: {len(paths)}")
    
    if paths:
        yolo_model.set_model(paths[0])

    return paths



# Setea yolo
def upload_yolo_model(path):
    print(f"Se cambia a modelo {path}")
    yolo_model.set_model(path)

# **Función que se ejecuta al cargar una imagen**


def on_image_load(image_path):
    try:
        #print("BLIP captioning started 👀")
        #caption = captioning_model.generate_caption(
           # image_path)  # Generar el caption usando el path
        caption=""
        #print("BLIP captioning finished")
        return caption, image_path, None  # Retornar el caption para que se muestre en el campo de texto y la ruta del archivo original
    except Exception as e:
        print(f"Error en la generación del caption: {e}")
        return "Error en la generación del caption"


# Construcción de la interfaz en Gradio
with gr.Blocks() as demo:
    gr.Markdown("# AI Impainter")
    gr.Markdown(
        "Con YOLOV8"
    )

    with gr.Row():
        img = gr.Image(label="Input Image", type="filepath")
        img_yolo = gr.Image(label="Yolo Image", type="pil")
        processed_img = gr.Image(label="Processed Mask", type="filepath", interactive=False)

    with gr.Row(equal_height=True):
        yolo_model_path = gr.Dropdown(choices=list_best_pt(), label="Modelos disponibles", scale=4)
        yolo_confidence = gr.Slider(
            minimum=0,
            maximum=1,
            value=0.1,
            step=0.01,
            label="Confianza",
            scale=1,
            interactive=True
        )

    with gr.Row():
        detect_button = gr.Button("Detectar Manchas")

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
            label="Negative prompt", 
            placeholder="Write negative prompt...", 
            value="Restore the original clean surface, natural texture and lighting. Remove all stains and blemishes in the masked areas, seamlessly blending with the surrounding. Maintain realistic details and consistency with the rest of the image.")

    with gr.Row():
        send_button = gr.Button("Impaint Image")

    with gr.Row():
        final_image = gr.Image(label="Final Output", type="filepath")

    gr.Markdown("---")
    gr.Markdown("## Resultados")

    with gr.Row():
        original_img = gr.Image(label="Original Image", type="filepath", interactive=False)
        impainted_img = gr.Image(label="Impainted Image", type="filepath", interactive=False)

    def generate_mask_with_yolo(image_path: str, confidence):
        try:
            print("YOLO detection started 🔍")
            yolo_image, boxes = yolo_model.get_bounding_box(confidence, image_path)
            print(f"YOLO detection has finished succesfully. {len(boxes)} boxes")

            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(
                    "No se pudo cargar la imagen. Verifica la ruta.")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            binary_mask = None
            # Generación de la máscara
            if (len(boxes)>0):
              print(f"SAM detection for {len(boxes)} box started 🔬")
              masks = segmentation_model.get_mask_by_bounding_boxes(boxes=boxes, image=image)
              print(f"SAM detection has finished successfully")

              # Mezclar multiples mascaras de SAM
              numpy_masks = [mask.cpu().numpy() for mask in masks]
              combined_mask = np.zeros_like(numpy_masks[0], dtype=bool)
              for m in numpy_masks:
                combined_mask = np.logical_or(combined_mask, m)

              # Generar mascara binaria
              binary_mask=generate_binary_mask(combined_mask)
              print("Refining generated mask with OpenCV 🖌") 
              refined_binary_mask = delete_irrelevant_detected_pixels(
                  binary_mask)
              without_irrelevant_pixels_mask = fill_little_spaces(
                  refined_binary_mask)
              dilated_mask = soften_contours(without_irrelevant_pixels_mask)
              blurred_mask = dilated_mask
              print("Image was refined successfully!")

              # Guardar máscara procesada
              processed_mask = Image.fromarray(blurred_mask, mode='L')
              processed_mask.save(RUTA_MASCARA)

              return yolo_image, RUTA_MASCARA
            else:
              return yolo_image, None
        except Exception as e:
            print(f"Error: {e}")
            return None        
    
    # **Reiniciar la máscara al cambiar de imagen**
    def reset_mask(image_path):
        delete_files([RUTA_MASCARA, RUTA_IMAGEN_FINAL])
        return None, None, None

    # **Procesar la imagen con la máscara y el texto de entrada**
    def process_final_image(original_image_path, mask_path, text, strength, guidance, negative_prompt):
        try:
            print("SD XL Impainting started 🎨")
            new_image = impainting_model.impaint(
                image_path=original_image_path, mask_path=mask_path, prompt=text, strength=strength, guidance=guidance, negative_prompt=negative_prompt)
            print("SD XL Impainting process finished")

            new_image.save(RUTA_IMAGEN_FINAL)
            return RUTA_IMAGEN_FINAL, RUTA_IMAGEN_FINAL
        except Exception as e:
            print(f"Error: {e}")
            return None
        
    def on_clear_processed_mask():
        delete_files([RUTA_MASCARA])
        return None


    # **Asignar eventos a la interfaz**
    img.change(on_image_load, inputs=[img], outputs=[text_input, original_img, impainted_img])
    yolo_model_path.change(fn=upload_yolo_model, inputs=yolo_model_path, outputs=None)
    detect_button.click(generate_mask_with_yolo, inputs=[img, yolo_confidence], outputs=[img_yolo, processed_img])
    processed_img.clear(on_clear_processed_mask, outputs=[processed_img])
    img.change(reset_mask, inputs=[img], outputs=[img_yolo, processed_img, final_image])
    send_button.click(process_final_image, inputs=[
                      img, processed_img, text_input, strength, guidance, negative_prompt], outputs=[final_image, impainted_img])
   

# **Limpiar archivos previos antes de lanzar la aplicación**
delete_files([RUTA_MASCARA, RUTA_IMAGEN_FINAL])

# **Lanzar la interfaz**
demo.launch(debug=True)
