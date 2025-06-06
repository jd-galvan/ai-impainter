import gradio as gr
import os
import random
from PIL import Image, ImageOps

# Ruta a la carpeta con las imágenes
CARPETA_IMAGENES = "/Users/josegalvan/Downloads/fotos_mejora2/fotos"

# Precargar ternas de imágenes


def precargar_imagenes(carpeta):
    archivos = os.listdir(carpeta)
    imagenes_base = sorted([
        f[:-4] for f in archivos
        if f.endswith(".jpg") and
        f[:-4] + "_RESTORED_UNet.png" in archivos and
        f[:-4] + "_RESTORED_YOLO+SAM.png" in archivos
    ])

    ternas = []
    for base in imagenes_base:
        try:
            original = Image.open(os.path.join(
                carpeta, base + ".jpg")).convert("RGB")
            original = ImageOps.exif_transpose(original)
            unet = Image.open(os.path.join(
                carpeta, base + "_RESTORED_UNet.png")).convert("RGB")
            yolo_sam = Image.open(os.path.join(
                carpeta, base + "_RESTORED_YOLO+SAM.png")).convert("RGB")
            ternas.append({
                "original": (original, base + ".jpg"),
                "restored": [
                    (unet, base + "_RESTORED_UNet.png"),
                    (yolo_sam, base + "_RESTORED_YOLO+SAM.png")
                ]
            })
        except Exception as e:
            print(f"Error cargando {base}: {e}")
    return ternas


imagenes_precargadas = precargar_imagenes(CARPETA_IMAGENES)
indice = 0


def mostrar_siguiente():
    global indice
    if not imagenes_precargadas:
        return [None] * 6

    trio = imagenes_precargadas[indice]
    indice = (indice + 1) % len(imagenes_precargadas)

    # Mezclar las imágenes restauradas
    restored = trio["restored"]
    random.shuffle(restored)

    return (
        trio["original"][0], trio["original"][1],
        restored[0][0], restored[0][1],
        restored[1][0], restored[1][1]
    )


# Crear interfaz
with gr.Blocks() as demo:
    gr.Markdown("## Comparador de Imágenes Restauradas")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Imagen Original")
            img1 = gr.Image(label="Original")
            name1 = gr.Markdown()
        with gr.Column():
            gr.Markdown("### Restauración 1")
            img2 = gr.Image()
            name2 = gr.Markdown()
            slider1a = gr.Slider(
                1, 10, step=1, label="Conservacion de identidad", interactive=True)
            slider1b = gr.Slider(
                1, 10, step=1, label="Desaparición de las manchas", interactive=True)
            slider1c = gr.Slider(
                1, 10, step=1, label="Reconstrucción coherente de zonas dañadas", interactive=True)
        with gr.Column():
            gr.Markdown("### Restauración 2")
            img3 = gr.Image()
            name3 = gr.Markdown()
            slider2a = gr.Slider(
                1, 10, step=1, label="Conservacion de identidad", interactive=True)
            slider2b = gr.Slider(
                1, 10, step=1, label="Desaparición de las manchas", interactive=True)
            slider2c = gr.Slider(
                1, 10, step=1, label="Reconstrucción coherente de zonas dañadas", interactive=True)
    with gr.Row():
        btn = gr.Button("Siguiente")

    btn.click(
        fn=mostrar_siguiente,
        outputs=[
            img1, name1,
            img2, name2,
            img3, name3
        ]
    )

demo.launch()
