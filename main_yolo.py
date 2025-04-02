import io
import cv2
import os
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
yolo_model = YOLOV8()

# Lista Yolos entrenado
def list_best_pt():
    paths = glob.glob("./runs/detect/train*/weights/best.pt")
    
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
        caption = captioning_model.generate_caption(
            image_path)  # Generar el caption usando el path
        return caption  # Retornar el caption para que se muestre en el campo de texto
    except Exception as e:
        print(f"Error en la generación del caption: {e}")
        return "Error en la generación del caption"


# Construcción de la interfaz en Gradio
with gr.Blocks() as demo:
    gr.Markdown("## AI Impainter")
    gr.Markdown(
        "Con YOLOV8"
    )

    with gr.Row():
        img = gr.Image(label="Input Image", type="filepath")
        img_yolo = gr.Image(label="Yolo Image", type="pil")
        processed_img = gr.Image(label="Processed Mask", type="filepath")

    with gr.Row(equal_height=True):
        yolo_model_path = gr.Dropdown(choices=list_best_pt(), label="Modelos disponibles")

    with gr.Row():
        chk_centric_pixel = gr.Checkbox(
          label="Usar pixeles céntricos",
          info="Si activas esta opción la máscara se generará dandole a SAM el pixel del centro del bounding box detectado por YOLO"
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
            label="Negative prompt", placeholder="Write negative prompt...")

    with gr.Row():
        send_button = gr.Button("Generate Final Image")

    with gr.Row():
        final_image = gr.Image(label="Final Output", type="filepath")

    # **Asignar la función para que se ejecute cuando la imagen se cargue**
    # img.change(on_image_load, inputs=[img], outputs=text_input)

    def generate_mask_with_yolo(image_path: str, checkbox_value):
        try:
            yolo_image, boxes = yolo_model.get_bounding_box(image_path)
            print(yolo_image) 
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(
                    "No se pudo cargar la imagen. Verifica la ruta.")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            binary_mask = None
            # Generación de la máscara
            for b in boxes:
              if checkbox_value:
                mask_image = segmentation_model.get_mask_by_pixel(
                  x=b[0] + ((b[2]-b[0])/2),
                  y=b[1] + (b[3]-b[1])/2,
                  image=image
                )
              else:
                mask_image = segmentation_model.get_mask_by_bounding_box(
                    x1=b[0],
                    y1=b[1],
                    x2=b[2],
                    y2=b[3],
                    image=image
                )
              mask = generate_binary_mask(mask_image)

              if binary_mask is not None:
                  binary_mask = np.maximum(binary_mask[:, :], mask)
              else:
                  binary_mask = mask.copy()
            
            refined_binary_mask = delete_irrelevant_detected_pixels(
                binary_mask)
            without_irrelevant_pixels_mask = fill_little_spaces(
                refined_binary_mask)
            dilated_mask = soften_contours(without_irrelevant_pixels_mask)
            blurred_mask = dilated_mask

            # Guardar máscara procesada
            processed_mask = Image.fromarray(blurred_mask, mode='L')
            processed_mask.save(RUTA_MASCARA)

            return yolo_image, RUTA_MASCARA
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
            new_image = impainting_model.impaint(
                image_path=original_image_path, mask_path=mask_path, prompt=text, strength=strength, guidance=guidance, negative_prompt=negative_prompt)
            new_image.save(RUTA_IMAGEN_FINAL)
            return RUTA_IMAGEN_FINAL
        except Exception as e:
            print(f"Error: {e}")
            return None
        
    def on_clear_processed_mask():
        delete_files([RUTA_MASCARA])
        return None


    # **Asignar eventos a la interfaz**
    yolo_model_path.change(fn=upload_yolo_model, inputs=yolo_model_path, outputs=None)
    detect_button.click(generate_mask_with_yolo, inputs=[img, chk_centric_pixel], outputs=[img_yolo, processed_img])
    processed_img.clear(on_clear_processed_mask, outputs=[processed_img])
    img.change(reset_mask, inputs=[img], outputs=[img_yolo, processed_img, final_image])
    send_button.click(process_final_image, inputs=[
                      img, processed_img, text_input, strength, guidance, negative_prompt], outputs=final_image)
   

# **Limpiar archivos previos antes de lanzar la aplicación**
delete_files([RUTA_MASCARA, RUTA_IMAGEN_FINAL])

# **Lanzar la interfaz**
demo.launch(debug=True)
