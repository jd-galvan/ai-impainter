import gradio as gr
import os
import random
from PIL import Image, ImageOps

# Ruta a la carpeta con las imágenes
CARPETA_IMAGENES = "/Users/josegalvan/Downloads/fotos_mejora2/fotos"

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

# Variables globales
imagenes_precargadas = precargar_imagenes(CARPETA_IMAGENES)
indice = 0
nombre_usuario = ""

def iniciar_evaluacion(nombre):
    global indice, nombre_usuario
    if not nombre.strip():
        return [
            gr.update(visible=True, value="Por favor, ingresa tu nombre"),  # error_msg
            gr.update(visible=True),  # nombre_input
            gr.update(visible=True),  # comenzar_btn
            *[gr.update(visible=False) for _ in range(21)]  # resto de componentes
        ]
    
    nombre_usuario = nombre.strip()
    indice = 1  # Iniciamos en 1 porque ya vamos a mostrar la primera imagen
    
    # Preparar primera imagen
    trio = imagenes_precargadas[0]
    restored = trio["restored"]
    random.shuffle(restored)
    
    return [
        gr.update(visible=False),  # error_msg
        gr.update(visible=False),  # nombre_input
        gr.update(visible=False),  # comenzar_btn
        gr.update(value=trio["original"][0], visible=True),  # img1
        gr.update(value=trio["original"][1], visible=True),  # name1
        gr.update(value=restored[0][0], visible=True),       # img2
        gr.update(value=restored[0][1], visible=True),       # name2
        gr.update(value=restored[1][0], visible=True),       # img3
        gr.update(value=restored[1][1], visible=True),       # name3
        gr.update(value=f"### Imagen Original 1/{len(imagenes_precargadas)}", visible=True),  # progress
        gr.update(value="", visible=False),  # gracias
        gr.update(visible=True),  # btn
        gr.update(visible=True),  # title
        gr.update(visible=True),  # markdown1
        gr.update(visible=True),  # markdown2
        gr.update(visible=True),  # slider1a
        gr.update(visible=True),  # slider1b
        gr.update(visible=True),  # slider1c
        gr.update(visible=True),  # markdown3
        gr.update(visible=True),  # slider2a
        gr.update(visible=True),  # slider2b
        gr.update(visible=True),  # slider2c
        gr.update(visible=True),  # preference
        gr.update(visible=False)  # reiniciar_btn
    ]

def mostrar_siguiente():
    global indice
    total = len(imagenes_precargadas)
    
    # Si no hay imágenes, retornar todo oculto
    if not imagenes_precargadas:
        return [
            gr.update(visible=False),  # error_msg
            gr.update(visible=True),   # nombre_input
            gr.update(visible=True),   # comenzar_btn
            *[gr.update(value=None, visible=False) for _ in range(21)]
        ]

    # Si estamos en la última imagen y presionamos siguiente
    if indice >= total:
        return [
            gr.update(visible=False),  # error_msg
            gr.update(visible=False),  # nombre_input
            gr.update(visible=False),  # comenzar_btn
            gr.update(value=None, visible=False),  # img1
            gr.update(value=None, visible=False),  # name1
            gr.update(value=None, visible=False),  # img2
            gr.update(value=None, visible=False),  # name2
            gr.update(value=None, visible=False),  # img3
            gr.update(value=None, visible=False),  # name3
            gr.update(value=None, visible=False),  # progress
            gr.update(value=f"\n\n# ¡Gracias por tus respuestas, {nombre_usuario}!", visible=True),  # gracias
            gr.update(visible=False),  # btn
            gr.update(visible=False),  # title
            gr.update(visible=False),  # markdown1
            gr.update(visible=False),  # markdown2
            gr.update(value=1, visible=False),  # slider1a
            gr.update(value=1, visible=False),  # slider1b
            gr.update(value=1, visible=False),  # slider1c
            gr.update(visible=False),  # markdown3
            gr.update(value=1, visible=False),  # slider2a
            gr.update(value=1, visible=False),  # slider2b
            gr.update(value=1, visible=False),  # slider2c
            gr.update(value=None, visible=False),  # preference
            gr.update(visible=True)    # reiniciar_btn
        ]

    # En cualquier otro caso, mostrar la imagen actual
    trio = imagenes_precargadas[indice]
    current_index = indice + 1

    # Mezclar las imágenes restauradas
    restored = trio["restored"]
    random.shuffle(restored)

    progress_text = f"### Imagen Original {current_index}/{total}"
    
    # Incrementar el índice para la próxima vez
    indice += 1

    return [
        gr.update(visible=False),  # error_msg
        gr.update(visible=False),  # nombre_input
        gr.update(visible=False),  # comenzar_btn
        gr.update(value=trio["original"][0], visible=True),  # img1
        gr.update(value=trio["original"][1], visible=True),  # name1
        gr.update(value=restored[0][0], visible=True),       # img2
        gr.update(value=restored[0][1], visible=True),       # name2
        gr.update(value=restored[1][0], visible=True),       # img3
        gr.update(value=restored[1][1], visible=True),       # name3
        gr.update(value=progress_text, visible=True),        # progress
        gr.update(value="", visible=False),                  # gracias
        gr.update(visible=True),                            # btn
        gr.update(visible=True),                            # title
        gr.update(visible=True),                            # markdown1
        gr.update(visible=True),                            # markdown2
        gr.update(value=1, visible=True),                   # slider1a
        gr.update(value=1, visible=True),                   # slider1b
        gr.update(value=1, visible=True),                   # slider1c
        gr.update(visible=True),                            # markdown3
        gr.update(value=1, visible=True),                   # slider2a
        gr.update(value=1, visible=True),                   # slider2b
        gr.update(value=1, visible=True),                   # slider2c
        gr.update(value=None, visible=True),                # preference
        gr.update(visible=False)                            # reiniciar_btn
    ]

# Crear interfaz
with gr.Blocks() as demo:
    title = gr.Markdown("## Comparador de Imágenes Restauradas")
    
    # Componentes de la pantalla inicial
    error_msg = gr.Markdown("", visible=False)
    nombre_input = gr.Textbox(label="Por favor, ingresa tu nombre", placeholder="Nombre")
    comenzar_btn = gr.Button("Comenzar Evaluación")

    with gr.Row():
        with gr.Column():
            with gr.Row():
                progress = gr.Markdown(f"### Imagen Original 1/{len(imagenes_precargadas)}", visible=False)
            img1 = gr.Image(label="Original", visible=False)
            name1 = gr.Markdown(visible=False)
            markdown1 = gr.Markdown("""
            **¿Cómo evaluar las restauraciones?**  
            - **Conservación de identidad**: 1 si las personas no se reconocen, 10 si se distingue perfectamente que las personas siguen siendo ellas. 
            - **Desaparición de las manchas**: 1 si la mancha sigue teniendo el mismo tamaño o ha aumentado, 10 si la mancha ha desaparecido, independientemente de la coherencia de las partes generadas.
            - **Reconstrucción coherente**: 1 si las partes reconstruidas no tienen nada que ver con el resto de la imagen o no son realistas, 10 si la imagen es coherente y no parece generada, independientemente de si mantiene o no manchas.
            """, visible=False)
        with gr.Column():
            markdown2 = gr.Markdown("### Restauración 1", visible=False)
            img2 = gr.Image(visible=False)
            name2 = gr.Markdown(visible=False)
            slider1a = gr.Slider(
                1, 10, step=1, label="Conservacion de identidad", interactive=True, visible=False)
            slider1b = gr.Slider(
                1, 10, step=1, label="Desaparición de las manchas", interactive=True, visible=False)
            slider1c = gr.Slider(
                1, 10, step=1, label="Reconstrucción coherente de zonas dañadas", interactive=True, visible=False)
        with gr.Column():
            markdown3 = gr.Markdown("### Restauración 2", visible=False)
            img3 = gr.Image(visible=False)
            name3 = gr.Markdown(visible=False)
            slider2a = gr.Slider(
                1, 10, step=1, label="Conservacion de identidad", interactive=True, visible=False)
            slider2b = gr.Slider(
                1, 10, step=1, label="Desaparición de las manchas", interactive=True, visible=False)
            slider2c = gr.Slider(
                1, 10, step=1, label="Reconstrucción coherente de zonas dañadas", interactive=True, visible=False)
    
    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=1):
            preference = gr.Radio(
                choices=["Restauración 1", "Restauración 2"],
                label="¿Cuál restauración te gustó más?",
                interactive=True,
                visible=False
            )
    
    with gr.Row():
        gracias = gr.Markdown("", visible=False)
    
    with gr.Row():
        btn = gr.Button("Siguiente", visible=False)
        reiniciar_btn = gr.Button("Realizar Nueva Evaluación", visible=False)

    # Eventos
    comenzar_btn.click(
        fn=iniciar_evaluacion,
        inputs=[nombre_input],
        outputs=[
            error_msg,
            nombre_input,
            comenzar_btn,
            img1, name1,
            img2, name2,
            img3, name3,
            progress,
            gracias,
            btn,
            title,
            markdown1,
            markdown2,
            slider1a,
            slider1b,
            slider1c,
            markdown3,
            slider2a,
            slider2b,
            slider2c,
            preference,
            reiniciar_btn
        ]
    )

    btn.click(
        fn=mostrar_siguiente,
        outputs=[
            error_msg,
            nombre_input,
            comenzar_btn,
            img1, name1,
            img2, name2,
            img3, name3,
            progress,
            gracias,
            btn,
            title,
            markdown1,
            markdown2,
            slider1a,
            slider1b,
            slider1c,
            markdown3,
            slider2a,
            slider2b,
            slider2c,
            preference,
            reiniciar_btn
        ]
    )

    reiniciar_btn.click(
        fn=lambda: iniciar_evaluacion(nombre_usuario),
        outputs=[
            error_msg,
            nombre_input,
            comenzar_btn,
            img1, name1,
            img2, name2,
            img3, name3,
            progress,
            gracias,
            btn,
            title,
            markdown1,
            markdown2,
            slider1a,
            slider1b,
            slider1c,
            markdown3,
            slider2a,
            slider2b,
            slider2c,
            preference,
            reiniciar_btn
        ]
    )

demo.launch()
