import gradio as gr
import os
import random
from PIL import Image, ImageOps
from datetime import datetime
import csv

# Ruta a la carpeta con las imágenes
CARPETA_IMAGENES = "/Users/josegalvan/Documents/Personal/UPV/salvem_les_fotos/restauracion_bench/benchmark"
# Carpeta para guardar los resultados
CARPETA_RESULTADOS = "resultados_benchmark"

# Asegurar que existe la carpeta de resultados
if not os.path.exists(CARPETA_RESULTADOS):
    os.makedirs(CARPETA_RESULTADOS)

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
archivo_respuestas = ""

def crear_archivo_respuestas(nombre):
    """Crea un archivo CSV para las respuestas del usuario."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"{nombre.replace(' ', '_')}_{timestamp}.csv"
    ruta_archivo = os.path.join(CARPETA_RESULTADOS, nombre_archivo)
    
    with open(ruta_archivo, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'imagen_original',
            'restauracion1_nombre',
            'restauracion1_identidad',
            'restauracion1_no_rostros',
            'restauracion1_manchas',
            'restauracion1_coherencia',
            'restauracion2_nombre',
            'restauracion2_identidad',
            'restauracion2_no_rostros',
            'restauracion2_manchas',
            'restauracion2_coherencia',
            'preferencia'
        ])
    
    return ruta_archivo

def encontrar_evaluacion_incompleta(nombre):
    """Busca si existe una evaluación incompleta para el usuario."""
    nombre_base = nombre.replace(' ', '_')
    if not os.path.exists(CARPETA_RESULTADOS):
        return None, 0
    
    # Buscar archivos que coincidan con el patrón del nombre y no estén completados
    archivos = [f for f in os.listdir(CARPETA_RESULTADOS) 
               if f.startswith(nombre_base) and 
               f.endswith('.csv') and 
               '_COMPLETED' not in f]
    
    if not archivos:
        return None, 0
    
    # Usar el archivo más reciente si hay varios
    archivo = sorted(archivos)[-1]
    ruta_archivo = os.path.join(CARPETA_RESULTADOS, archivo)
    
    # Contar cuántas imágenes ya se evaluaron
    with open(ruta_archivo, 'r') as f:
        # Restar 1 para no contar el encabezado
        imagenes_evaluadas = sum(1 for _ in f) - 1
    
    return ruta_archivo, imagenes_evaluadas

def iniciar_evaluacion(nombre):
    global indice, nombre_usuario, archivo_respuestas
    if not nombre.strip():
        return [
            gr.update(visible=True, value="Por favor, ingresa tu nombre"),  # error_msg
            gr.update(visible=True),  # nombre_input
            gr.update(visible=True),  # comenzar_btn
            gr.update(value=None, visible=False),  # img1
            gr.update(value=None, visible=False),  # name1
            gr.update(value=None, visible=False),  # img2
            gr.update(value=None, visible=False),  # name2
            gr.update(value=None, visible=False),  # img3
            gr.update(value=None, visible=False),  # name3
            gr.update(value=None, visible=False),  # progress
            gr.update(value="", visible=False),  # gracias
            gr.update(visible=False),  # btn
            gr.update(visible=False),  # title
            gr.update(visible=False),  # markdown1
            gr.update(visible=False),  # markdown2
            gr.update(value=1, visible=False),  # slider1a
            gr.update(value=False, visible=False),  # checkbox1
            gr.update(value=1, visible=False),  # slider1b
            gr.update(value=1, visible=False),  # slider1c
            gr.update(visible=False),  # markdown3
            gr.update(value=1, visible=False),  # slider2a
            gr.update(value=False, visible=False),  # checkbox2
            gr.update(value=1, visible=False),  # slider2b
            gr.update(value=1, visible=False),  # slider2c
            gr.update(value=None, visible=False),  # preference
            gr.update(visible=False)  # reiniciar_btn
        ]
    
    nombre_usuario = nombre.strip()
    
    # Buscar si existe una evaluación incompleta
    evaluacion_existente, imagenes_evaluadas = encontrar_evaluacion_incompleta(nombre_usuario)
    
    if evaluacion_existente:
        archivo_respuestas = evaluacion_existente
        indice = imagenes_evaluadas  # Comenzar desde donde se quedó
    else:
        archivo_respuestas = crear_archivo_respuestas(nombre_usuario)
        indice = 0
    
    # Preparar imagen actual
    trio = imagenes_precargadas[indice]
    restored = trio["restored"]
    random.shuffle(restored)
    
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
        gr.update(value=f"### Imagen Original {indice}/{len(imagenes_precargadas)}", visible=True),  # progress
        gr.update(value="", visible=False),  # gracias
        gr.update(visible=True),  # btn
        gr.update(visible=True),  # title
        gr.update(visible=True),  # markdown1
        gr.update(visible=True),  # markdown2
        gr.update(value=1, visible=True),  # slider1a
        gr.update(value=False, visible=True),  # checkbox1
        gr.update(value=1, visible=True),  # slider1b
        gr.update(value=1, visible=True),  # slider1c
        gr.update(visible=True),  # markdown3
        gr.update(value=1, visible=True),  # slider2a
        gr.update(value=False, visible=True),  # checkbox2
        gr.update(value=1, visible=True),  # slider2b
        gr.update(value=1, visible=True),  # slider2c
        gr.update(value=None, visible=True),  # preference
        gr.update(visible=False)  # reiniciar_btn
    ]

def guardar_respuestas(imagen_original, rest1_nombre, rest1_identidad, rest1_no_rostros, rest1_manchas, rest1_coherencia,
                      rest2_nombre, rest2_identidad, rest2_no_rostros, rest2_manchas, rest2_coherencia, preferencia):
    """Guarda las respuestas de la evaluación actual en el archivo CSV."""
    with open(archivo_respuestas, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            imagen_original,
            rest1_nombre,
            rest1_identidad,
            rest1_no_rostros,
            rest1_manchas,
            rest1_coherencia,
            rest2_nombre,
            rest2_identidad,
            rest2_no_rostros,
            rest2_manchas,
            rest2_coherencia,
            preferencia
        ])

def mostrar_siguiente(slider1a, checkbox1, slider1b, slider1c,
                     slider2a, checkbox2, slider2b, slider2c,
                     preference):
    global indice, archivo_respuestas
    total = len(imagenes_precargadas)
    
    # Guardar respuestas de la imagen actual
    if indice > 0:  # No guardamos al inicio
        imagen_actual = imagenes_precargadas[indice - 1]
        guardar_respuestas(
            imagen_actual["original"][1],  # nombre imagen original
            imagen_actual["restored"][0][1],  # nombre restauración 1
            slider1a,
            checkbox1,
            slider1b,
            slider1c,
            imagen_actual["restored"][1][1],  # nombre restauración 2
            slider2a,
            checkbox2,
            slider2b,
            slider2c,
            preference
        )

    # Si estamos en la última imagen y presionamos siguiente
    if indice >= total:
        # Renombrar el archivo añadiendo COMPLETED
        if os.path.exists(archivo_respuestas):
            nuevo_nombre = archivo_respuestas.replace('.csv', '_COMPLETED.csv')
            os.rename(archivo_respuestas, nuevo_nombre)
            archivo_respuestas = nuevo_nombre

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
            gr.update(value=False, visible=False),  # checkbox1
            gr.update(value=1, visible=False),  # slider1b
            gr.update(value=1, visible=False),  # slider1c
            gr.update(visible=False),  # markdown3
            gr.update(value=1, visible=False),  # slider2a
            gr.update(value=False, visible=False),  # checkbox2
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
        gr.update(value=False, visible=True),               # checkbox1
        gr.update(value=1, visible=True),                   # slider1b
        gr.update(value=1, visible=True),                   # slider1c
        gr.update(visible=True),                            # markdown3
        gr.update(value=1, visible=True),                   # slider2a
        gr.update(value=False, visible=True),               # checkbox2
        gr.update(value=1, visible=True),                   # slider2b
        gr.update(value=1, visible=True),                   # slider2c
        gr.update(value=None, visible=True),                # preference
        gr.update(visible=False)                            # reiniciar_btn
    ]

def toggle_slider1(checkbox_value):
    return gr.update(value=1, interactive=not checkbox_value)

def toggle_slider2(checkbox_value):
    return gr.update(value=1, interactive=not checkbox_value)

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
            with gr.Row():
                slider1a = gr.Slider(1, 10, step=1, label="Conservacion de identidad", interactive=True, visible=False)
                checkbox1 = gr.Checkbox(label="No hay rostros", visible=False)
            slider1b = gr.Slider(1, 10, step=1, label="Desaparición de las manchas", interactive=True, visible=False)
            slider1c = gr.Slider(1, 10, step=1, label="Reconstrucción coherente de zonas dañadas", interactive=True, visible=False)
        with gr.Column():
            markdown3 = gr.Markdown("### Restauración 2", visible=False)
            img3 = gr.Image(visible=False)
            name3 = gr.Markdown(visible=False)
            with gr.Row():
                slider2a = gr.Slider(1, 10, step=1, label="Conservacion de identidad", interactive=True, visible=False)
                checkbox2 = gr.Checkbox(label="No hay rostros", visible=False)
            slider2b = gr.Slider(1, 10, step=1, label="Desaparición de las manchas", interactive=True, visible=False)
            slider2c = gr.Slider(1, 10, step=1, label="Reconstrucción coherente de zonas dañadas", interactive=True, visible=False)
    
    gr.HTML("<hr>")

    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=2):
            preference = gr.Radio(
                choices=["Restauración 1", "Restauración 2", "Ambas son igual de buenas"],
                label="¿Cuál restauración te gustó más?",
                interactive=True,
                visible=False
            )
    
    with gr.Row():
        gracias = gr.Markdown("", visible=False)
    
    with gr.Row():
        btn = gr.Button("Siguiente", visible=False)
        reiniciar_btn = gr.Button("Realizar Nueva Evaluación", visible=False)

    # Eventos de los checkboxes
    checkbox1.change(fn=toggle_slider1, inputs=[checkbox1], outputs=[slider1a])
    checkbox2.change(fn=toggle_slider2, inputs=[checkbox2], outputs=[slider2a])

    # Eventos principales
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
            checkbox1,
            slider1b,
            slider1c,
            markdown3,
            slider2a,
            checkbox2,
            slider2b,
            slider2c,
            preference,
            reiniciar_btn
        ]
    )

    btn.click(
        fn=mostrar_siguiente,
        inputs=[
            slider1a, checkbox1, slider1b, slider1c,
            slider2a, checkbox2, slider2b, slider2c,
            preference
        ],
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
            checkbox1,
            slider1b,
            slider1c,
            markdown3,
            slider2a,
            checkbox2,
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
            checkbox1,
            slider1b,
            slider1c,
            markdown3,
            slider2a,
            checkbox2,
            slider2b,
            slider2c,
            preference,
            reiniciar_btn
        ]
    )

demo.launch()
