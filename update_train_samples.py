import io
import cv2
import os
import gradio as gr
import glob
import torch
import torchvision.transforms as transforms
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import pandas as pd
from libs.inpainter import SDImpainting
from models import UNet
from libs.metrics import reMOVE
from utils import (
    generate_binary_mask,
    delete_irrelevant_detected_pixels,
    fill_little_spaces,
    soften_contours,
    blur_mask,
    delete_files
)
import gc


print("All packages imported")
# Cargar variables de entorno
load_dotenv()
images = sorted(glob.glob("RUTA"))
i = -1

# Rutas de archivos generados
RUTA_MASCARA = "processed_mask.png"
RUTA_IMAGEN_FINAL = "final_output.png"
transformations = transforms.Compose([transforms.Resize((1024, 1024)),transforms.ToTensor()])

# Configuracion del dispositivo para modelos
DEVICE = "cuda:1"
DEVICE_UNET = "cuda:1"
print(f"DEVICE {DEVICE}")

# Cargar modelos
segmentation_model = UNet(n_class = 1)
impainting_model = SDImpainting(DEVICE)
remove = reMOVE(device = DEVICE_UNET)

df_smudges = pd.DataFrame({
            "type": ["original", "restored", "ratio"],
            "value": [0,0,0]})
            
df_remove = pd.DataFrame({
            "type": ["clean_mask-dirty_mask", "clean_mask-original", "dirty_mask-original"],
            "value": [0,0,0]})

# Funcion para sacar todos los modelos de segmentacion

def list_models():
    paths = glob.glob("../output/unet*.pth")
    return paths
    
def list_images():
    masks = sorted(glob.glob('../segmentation/masks/*.*'))
    masks = masks[round(0.8*len(masks)):]
    return masks
    
    
def make_metrics_plot_smudges(original_image_path, restored_image_path, predicted_mask, thr):
    try:
        pred_image_path = apply_fill_little_spaces(predicted_mask, "pred")
        mask = transforms.ToTensor()(Image.open(predicted_mask).convert("L"))
        mask_dilatation = transforms.ToTensor()(Image.open(pred_image_path).convert("L"))
        torch.cuda.empty_cache()
        gc.collect()
        
        original_image = Image.open(original_image_path[:-3].replace("masks", "images")+"jpg").convert('RGB')
        restored_image = Image.open(restored_image_path).convert('RGB')
        image_tensor = transformations(restored_image).reshape((1, 3, 1024, 1024)).to(DEVICE_UNET)
        global segmentation_model  
        segmentation_model = segmentation_model.to(DEVICE_UNET)
        restored_mask = segmentation_model(image_tensor)
        restored_mask = torch.sigmoid(restored_mask)
        restored_mask = (restored_mask >= thr).float()
        clean_original = float(torch.sum(mask>0)/torch.sum(mask>=0))
        clean_restored = float(torch.sum(restored_mask>0)/torch.sum(restored_mask>=0))
        df_smudges = pd.DataFrame({
            "type": ["original", "restored", "ratio"],
            "value": [clean_original,clean_restored,clean_restored/clean_original]})
        return gr.BarPlot(
                  df_smudges,
                  x="type",
                  y="value"
                )
    except Exception as e:
        print(f"Error: {e}")
        return None
        
def make_metrics_plot_remove(original_image_path, restored_image_path, predicted_mask):
    try:
        torch.cuda.empty_cache()
        gc.collect()
        remove_metrics = remove(predicted_mask, original_image_path, restored_image_path)
        df_remove = pd.DataFrame({
            "type": ["clean_mask-dirty_mask", "clean_mask-original", "dirty_mask-original"],
            "value": [remove_metrics["clean_mask-dirty_mask"], remove_metrics["clean_mask-original"], remove_metrics["dirty_mask-original"]]})
        return gr.BarPlot(
                  df_remove,
                  x="type",
                  y="value"
                )
    except Exception as e:
        print(f"Error: {e}")
        return None

# Funcion que se ejecuta al cargar una imagen


        
def on_image_load_image(image_path):
      image = Image.open(image_path[:-3].replace("masks", "images")+"jpg").convert('RGB')
      image = transformations(image)
      image = transforms.functional.to_pil_image(image).convert("RGB")
      image.save(f'image.jpg')
      return 'image.jpg'
        
def on_image_load_target(image_path):
    real_mask = Image.open(image_path).convert('RGB')
    real_mask = transformations(real_mask)
    real_mask = transforms.functional.to_pil_image(real_mask).convert("RGB")
    real_mask.save(f'target_mask.jpg')
    return 'target_mask.jpg'
        
def on_image_load_pred(image_path, thr):
    image = Image.open(image_path[:-3].replace("masks", "images")+"jpg").convert('RGB')
    torch.cuda.empty_cache()
    gc.collect()
    image_tensor = transformations(image).reshape((1, 3, 1024, 1024)).to(DEVICE_UNET)
    global segmentation_model  
    segmentation_model = segmentation_model.to(DEVICE_UNET)
    mask = segmentation_model(image_tensor)
    mask = torch.sigmoid(mask)
    mask_bin = (mask >= thr).float()
    mask = transforms.functional.to_pil_image(mask_bin[0]).convert("RGB")
    mask.save(f'pred_mask.jpg')
    return 'pred_mask.jpg'
    
def on_image_load_pred2(image_path, thr):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transformations(image).reshape((1, 3, 1024, 1024)).to(DEVICE_UNET)
    global segmentation_model  
    segmentation_model = segmentation_model.to(DEVICE_UNET)
    mask = segmentation_model(image_tensor)
    mask = torch.sigmoid(mask)
    mask_bin = (mask >= thr).float()
    mask = transforms.functional.to_pil_image(mask_bin[0]).convert("RGB")
    mask.save(f'pred_mask.jpg')
    return 'pred_mask.jpg'
    

def next_image_unet(thr):
    # Load images
    global images
    global i
    i += 1
    image_path = images[i]
    image = Image.open(image_path)
    image = transformations(image)
    image = transforms.functional.to_pil_image(image).convert("RGB")
    image.save(f'image.jpg')
    image = cv2.imread('image.jpg')
    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)

    # Binarize masks
    _, pred_mask_bin = cv2.threshold(pred_mask, int(255*thr), 255, cv2.THRESH_BINARY)

    # Identify false positives
    false_positive_mask_3ch = cv2.merge([pred_mask_bin]*3)

    # Fade the full image
    faded_image = cv2.addWeighted(image, 0.4, np.full_like(image, 255), 0.6, 0)

    # Create a red-tinted overlay for false positives
    red_overlay = image.copy()
    red_overlay[:, :, 0] = 0    # Zero out blue channel
    red_overlay[:, :, 1] = 0    # Zero out green channel
    red_overlay[:, :, 2] = 255  # Max red

    # Blend red overlay with original image in FP regions
    highlighted_fp = cv2.addWeighted(image, 0.3, red_overlay, 0.7, 0)

    # Combine: red-highlighted FP + faded elsewhere
    output_image = np.where(false_positive_mask_3ch == 255, highlighted_fp, faded_image)

    # Save result
    final_path = "false_positives_highlighted_red.png"
    cv2.imwrite(final_path, output_image)

    return final_path

    

def extract_false_negatives(image_path, thr):
    # Load images
    image = Image.open(image_path)
    image = transformations(image)
    print(image.size())
    print()
    print()
    image = transforms.functional.to_pil_image(image).convert("RGB")
    image.save(f'image.jpg')
    image = cv2.imread('image.jpg')
    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)

    # Binarize masks
    _, pred_mask_bin = cv2.threshold(pred_mask, 127, 255, cv2.THRESH_BINARY)

    # Identify false negatives: in target, missing in prediction
    false_negative_mask_3ch = cv2.merge([pred_mask_bin]*3)

    # Fade the entire image
    faded_image = cv2.addWeighted(image, 0.4, np.full_like(image, 255), 0.6, 0)

    # Create blue-tinted overlay
    blue_overlay = image.copy()
    blue_overlay[:, :, 0] = 255  # Max blue
    blue_overlay[:, :, 1] = 0    # Zero green
    blue_overlay[:, :, 2] = 0    # Zero red

    # Blend overlay with image in FN areas
    highlighted_fn = cv2.addWeighted(image, 0.3, blue_overlay, 0.7, 0)

    # Combine: blue-highlighted FN + faded elsewhere
    output_image = np.where(false_negative_mask_3ch == 255, highlighted_fn, faded_image)

    # Save result
    final_path = "false_negatives_highlighted_blue.png"
    cv2.imwrite(final_path, output_image)

    return final_path



# Funcion para aplicar fill_little_spaces al reenviar de la 3era a la 4ta imagen


def apply_fill_little_spaces(image_path, mask_type):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("No se pudo cargar la imagen.")
        processed = delete_irrelevant_detected_pixels(image)
        processed = fill_little_spaces(processed)
        processed = soften_contours(processed)
        output_path = f"softened_contours_{mask_type}.png"
        cv2.imwrite(output_path, processed)
        return output_path
    except Exception as e:
        print(f"Error en apply_fill_little_spaces: {e}")
        return None

# Procesar la imagen final usando la imagen original y la quinta imagen
def process_final_image_target(original_image_path, target_image_path, text, strength, guidance, negative_prompt):
    try:
        target_image_path = apply_fill_little_spaces(target_image_path, "target")
        torch.cuda.empty_cache()
        gc.collect()
        new_image = impainting_model.impaint(
            image_path=original_image_path,
            mask_path=target_image_path,
            prompt="",
            strength=strength,
            guidance=guidance,
            negative_prompt=negative_prompt
        )
        new_image.save("final_output_target.png")
        return "final_output_target.png"
    except Exception as e:
        print(f"Error: {e}")
        return None
        
        
        
def process_final_image_pred(original_image_path, pred_image_path, text, strength, guidance, negative_prompt):
    try:
        pred_image_path = apply_fill_little_spaces(pred_image_path, "pred")
        torch.cuda.empty_cache()
        gc.collect()
        new_image = impainting_model.impaint(
            image_path=original_image_path,
            mask_path=pred_image_path,
            prompt="",
            strength=strength,
            guidance=guidance,
            negative_prompt=negative_prompt
        )
        new_image.save("final_output_pred.png")
        return "final_output_pred.png"
    except Exception as e:
        print(f"Error: {e}")
        return None

def upload_model(path):
    print(f"Se cambia a modelo {path}")  
    
    torch.cuda.empty_cache()
    gc.collect()
    global segmentation_model  
    segmentation_model.load_state_dict(torch.load(f"{path}", weights_only=True))
    segmentation_model= segmentation_model.to(DEVICE_UNET)

# Construccion de la interfaz en Gradio
with gr.Blocks() as demo:

    # Modelo para segmentar
    with gr.Row():
        next_button = gr.Button("Next")
        threshold = gr.Slider(minimum=0.0, maximum=1.0,
                             value=0.5, label="Threshold", interactive=True)
        
        
    # Fila 1: Imagen de entrada, mascara real y mascara predicha
    with gr.Row():
        img = gr.Image(label="Input Image", type="filepath")
        unet = gr.Image(label="UNET", type="filepath")
        yolo = gr.Image(label="YOLO", type="filepath")
    
        
    # Fila 2: False positives y false negatives
    with gr.Row():
        unet_button = gr.Button("Download UNET mask")
        yolo_button = gr.Button("Download YOLO mask")

    # Al cargar la imagen se genera el caption en el textbox
    next_button.click(next_image_unet, inputs=[], outputs=unet)
    next_button.click(next_image_yolo, inputs=[], outputs=yolo)
    image_path.change(on_image_load_image, inputs=[image_path], outputs=img_real)
    image_path.change(on_image_load_target, inputs=[image_path], outputs=target)
    image_path.change(on_image_load_pred, inputs=[image_path, threshold], outputs=pred)
    mask_button.click(on_image_load_pred2, inputs=[img, threshold], outputs=pred)
    model_path.change(fn=upload_model, inputs=model_path, outputs=None)


    # Reiniciar la mascara al cambiar de imagen
    def reset_mask(image_path):
        delete_files([RUTA_MASCARA, RUTA_IMAGEN_FINAL])
        return None

    # Asignar eventos a la interfaz
    #img.change(reset_mask, inputs=[img], outputs=None)
    fp_button.click(extract_false_positives, inputs=[
                      img, target, pred, threshold], outputs=fp)
    fn_button.click(extract_false_negatives, inputs=[
                      img, target, pred, threshold], outputs=fn)

    # Boton para generar la imagen final usando la imagen original y las mascaras
    send_button_target.click(process_final_image_target, inputs=[
                      img, target, text_input, strength, guidance, negative_prompt], outputs=img_target)
    
    send_button_pred.click(process_final_image_pred, inputs=[
                      img, pred, text_input, strength, guidance, negative_prompt], outputs=img_pred)

# Limpiar archivos previos antes de lanzar la aplicacion
delete_files([RUTA_MASCARA, RUTA_IMAGEN_FINAL])

# Lanzar la interfaz
demo.launch(debug=True)