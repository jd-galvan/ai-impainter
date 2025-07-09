import gradio as gr
import os
import random
from PIL import Image, ImageOps
from datetime import datetime
import csv

# Ruta a la carpeta con las imágenes
CARPETA_IMAGENES = "/home/salvem/benchmarkv2"
# Carpeta para guardar los resultados
CARPETA_RESULTADOS = "resultados_benchmark"

# Ensure the results folder exists
if not os.path.exists(CARPETA_RESULTADOS):
    os.makedirs(CARPETA_RESULTADOS)

def precargar_imagenes(carpeta):
    archivos = os.listdir(carpeta)
    imagenes_base = sorted([
        f[:-4] for f in archivos
        if f.endswith(".jpg") and
        f[:-4] + "_RESTORED_UNet.png" in archivos and
        f[:-4] + "_RESTORED_YOLO+SAM.png" in archivos and
        f[:-4] + "_RESTORED_SegFormer.png" in archivos
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
            segformer = Image.open(os.path.join(
                carpeta, base + "_RESTORED_SegFormer.png")).convert("RGB")
            ternas.append({
                "original": (original, base + ".jpg"),
                "restored": [
                    (unet, base + "_RESTORED_UNet.png"),
                    (yolo_sam, base + "_RESTORED_YOLO+SAM.png"),
                    (segformer, base + "_RESTORED_SegFormer.png")
                ]
            })
        except Exception as e:
            print(f"Error cargando {base}: {e}")
    return ternas

# Variables globales
# imagenes_precargadas = precargar_imagenes(CARPETA_IMAGENES)
# indice = 0
# nombre_usuario = ""
# archivo_respuestas = ""

imagenes_precargadas = precargar_imagenes(CARPETA_IMAGENES)

# Estado inicial para cada usuario
initial_state = {
    "indice": 0,
    "nombre_usuario": "",
    "archivo_respuestas": ""
}

def crear_archivo_respuestas(nombre):
    """Creates a CSV file for the user's responses."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"{nombre.replace(' ', '_')}_{timestamp}.csv"
    ruta_archivo = os.path.join(CARPETA_RESULTADOS, nombre_archivo)
    
    with open(ruta_archivo, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'original_image',
            'no_faces',
            'restoration1_name',
            'restoration1_identity',
            'restoration1_stains',
            'restoration1_coherence',
            'restoration2_name',
            'restoration2_identity',
            'restoration2_stains',
            'restoration2_coherence',
            'restoration3_name',
            'restoration3_identity',
            'restoration3_stains',
            'restoration3_coherence',
            'general_comment',
            'preference'
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

def iniciar_evaluacion(nombre, state):
    if not nombre.strip():
        return [
            gr.update(visible=True, value="Please enter your name"),  # error_msg
            gr.update(visible=True),  # nombre_input
            gr.update(visible=True),  # comenzar_btn
            gr.update(visible=True),  # consent_accordion
            gr.update(visible=True),  # aceptar_consentimiento_checkbox
            gr.update(value=None, visible=False),  # img1
            gr.update(value=None, visible=False),  # name1
            gr.update(value=None, visible=False),  # img2
            gr.update(value=None, visible=False),  # name2
            gr.update(value=None, visible=False),  # img3
            gr.update(value=None, visible=False),  # name3
            gr.update(value=None, visible=False),  # img4
            gr.update(value=None, visible=False),  # name4
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
            gr.update(value=1, visible=False),  # slider2b
            gr.update(value=1, visible=False),  # slider2c
            gr.update(visible=False),  # markdown4
            gr.update(value=1, visible=False),  # slider3a
            gr.update(value=1, visible=False),  # slider3b
            gr.update(value=1, visible=False),  # slider3c
            gr.update(value="", visible=True),  # comentario_general
            gr.update(visible=False),  # ranking_instruction
            gr.update(value=None, visible=False),  # ranking1
            gr.update(value=None, visible=False),  # ranking2
            gr.update(value=None, visible=False),  # ranking3
            gr.update(visible=False),  # reiniciar_btn
            state
        ]
    state["nombre_usuario"] = nombre.strip()
    evaluacion_existente, imagenes_evaluadas = encontrar_evaluacion_incompleta(state["nombre_usuario"])
    if evaluacion_existente:
        state["archivo_respuestas"] = evaluacion_existente
        state["indice"] = imagenes_evaluadas  # Start from where left off
    else:
        state["archivo_respuestas"] = crear_archivo_respuestas(state["nombre_usuario"])
        state["indice"] = 0
    trio = imagenes_precargadas[state["indice"]]
    restored = trio["restored"]
    random.shuffle(restored)
    state["indice"] += 1
    return [
        gr.update(visible=False),  # error_msg
        gr.update(visible=False),  # nombre_input
        gr.update(visible=False),  # comenzar_btn
        gr.update(visible=False),  # consent_accordion
        gr.update(visible=False),  # aceptar_consentimiento_checkbox
        gr.update(value=trio["original"][0], visible=True),  # img1
        gr.update(value=trio["original"][1], visible=False),  # name1
        gr.update(value=restored[0][0], visible=True),       # img2
        gr.update(value=restored[0][1], visible=False),       # name2
        gr.update(value=restored[1][0], visible=True),       # img3
        gr.update(value=restored[1][1], visible=False),       # name3
        gr.update(value=restored[2][0], visible=True),       # img4
        gr.update(value=restored[2][1], visible=False),       # name4
        gr.update(value=f"### Original Image {state['indice']}/{len(imagenes_precargadas)}", visible=True),  # progress
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
        gr.update(value=1, visible=True),  # slider2b
        gr.update(value=1, visible=True),  # slider2c
        gr.update(visible=True),  # markdown4
        gr.update(value=1, visible=True),  # slider3a
        gr.update(value=1, visible=True),  # slider3b
        gr.update(value=1, visible=True),  # slider3c
        gr.update(value="", visible=True),  # comentario_general
        gr.update(visible=True),  # ranking_instruction
        gr.update(value=None, visible=True),  # ranking1
        gr.update(value=None, visible=True),  # ranking2
        gr.update(value=None, visible=True),  # ranking3
        gr.update(visible=False),  # reiniciar_btn
        state
    ]

def guardar_respuestas(imagen_original, no_rostros,
                      rest1_nombre, rest1_identidad, rest1_manchas, rest1_coherencia,
                      rest2_nombre, rest2_identidad, rest2_manchas, rest2_coherencia,
                      rest3_nombre, rest3_identidad, rest3_manchas, rest3_coherencia,
                      comentario_general,
                      ranking,
                      state):
    """Guarda las respuestas de la evaluación actual en el archivo CSV."""
    with open(state["archivo_respuestas"], 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            imagen_original,
            no_rostros,
            rest1_nombre,
            rest1_identidad,
            rest1_manchas,
            rest1_coherencia,
            rest2_nombre,
            rest2_identidad,
            rest2_manchas,
            rest2_coherencia,
            rest3_nombre,
            rest3_identidad,
            rest3_manchas,
            rest3_coherencia,
            comentario_general,
            ",".join(ranking) if ranking else ""
        ])

def mostrar_siguiente(slider1a, checkbox1, slider1b, slider1c,
                     slider2a, slider2b, slider2c,
                     slider3a, slider3b, slider3c,
                     comentario_general,
                     ranking1, ranking2, ranking3,
                     state):
    total = len(imagenes_precargadas)

    # Validate that all rankings are selected
    if not (ranking1 and ranking2 and ranking3):
        return [
            gr.update(visible=True, value="❌ Please select your 1st, 2nd, and 3rd preference before continuing."),  # error_msg
            gr.update(visible=False),  # nombre_input
            gr.update(visible=False),  # comenzar_btn
            gr.update(),  # img1
            gr.update(),  # name1
            gr.update(),  # img2
            gr.update(),  # name2
            gr.update(),  # img3
            gr.update(),  # name3
            gr.update(),  # img4
            gr.update(),  # name4
            gr.update(),  # progress
            gr.update(),  # gracias
            gr.update(),  # btn
            gr.update(),  # title
            gr.update(),  # markdown1
            gr.update(),  # markdown2
            gr.update(),  # slider1a
            gr.update(),  # checkbox1
            gr.update(),  # slider1b
            gr.update(),  # slider1c
            gr.update(),  # markdown3
            gr.update(),  # slider2a
            gr.update(),  # slider2b
            gr.update(),  # slider2c
            gr.update(),  # markdown4
            gr.update(),  # slider3a
            gr.update(),  # slider3b
            gr.update(),  # slider3c
            gr.update(),  # comentario_general
            gr.update(),  # ranking_instruction
            gr.update(),  # ranking1
            gr.update(),  # ranking2
            gr.update(),  # ranking3
            gr.update(),   # reiniciar_btn
            state
        ]
    
    # Guardar respuestas de la imagen actual
    if state["indice"] > 0:  # No guardamos al inicio
        imagen_actual = imagenes_precargadas[state["indice"] - 1]
        guardar_respuestas(
            imagen_actual["original"][1],  # nombre imagen original
            checkbox1,
            imagen_actual["restored"][0][1],  # nombre restauración 1
            slider1a,
            slider1b,
            slider1c,
            imagen_actual["restored"][1][1],  # nombre restauración 2
            slider2a,
            slider2b,
            slider2c,
            imagen_actual["restored"][2][1],  # nombre restauración 3
            slider3a,
            slider3b,
            slider3c,
            comentario_general,
            [ranking1, ranking2, ranking3],
            state
        )

    # If we are on the last image and press next
    if state["indice"] >= total:
        # Rename the file adding COMPLETED
        if os.path.exists(state["archivo_respuestas"]):
            nuevo_nombre = state["archivo_respuestas"].replace('.csv', '_COMPLETED.csv')
            os.rename(state["archivo_respuestas"], nuevo_nombre)
            state["archivo_respuestas"] = nuevo_nombre

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
            gr.update(value=None, visible=False),  # img4
            gr.update(value=None, visible=False),  # name4
            gr.update(value=None, visible=False),  # progress
            gr.update(value=f"\n\n# Thank you for your responses, {state['nombre_usuario']}!", visible=True),  # gracias
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
            gr.update(value=1, visible=False),  # slider2b
            gr.update(value=1, visible=False),  # slider2c
            gr.update(visible=False),  # markdown4
            gr.update(value=1, visible=False),  # slider3a
            gr.update(value=1, visible=False),  # slider3b
            gr.update(value=1, visible=False),  # slider3c
            gr.update(value="", visible=False),  # comentario_general
            gr.update(visible=False),  # ranking_instruction
            gr.update(value=None, visible=False),  # ranking1
            gr.update(value=None, visible=False),  # ranking2
            gr.update(value=None, visible=False),  # ranking3
            gr.update(visible=False),    # reiniciar_btn
            state
        ]

    # En cualquier otro caso, mostrar la imagen actual
    trio = imagenes_precargadas[state["indice"]]
    current_index = state["indice"] + 1

    # Mezclar las imágenes restauradas
    restored = trio["restored"]
    random.shuffle(restored)

    progress_text = f"### Original Image {current_index}/{total}"
    
    # Incrementar el índice para la próxima vez
    state["indice"] += 1

    return [
        gr.update(visible=False),  # error_msg
        gr.update(visible=False),  # nombre_input
        gr.update(visible=False),  # comenzar_btn
        gr.update(value=trio["original"][0], visible=True),  # img1
        gr.update(value=trio["original"][1], visible=False),  # name1
        gr.update(value=restored[0][0], visible=True),       # img2
        gr.update(value=restored[0][1], visible=False),       # name2
        gr.update(value=restored[1][0], visible=True),       # img3
        gr.update(value=restored[1][1], visible=False),       # name3
        gr.update(value=restored[2][0], visible=True),       # img4
        gr.update(value=restored[2][1], visible=False),       # name4
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
        gr.update(value=1, visible=True),                   # slider2b
        gr.update(value=1, visible=True),                   # slider2c
        gr.update(visible=True),                            # markdown4
        gr.update(value=1, visible=True),                   # slider3a
        gr.update(value=1, visible=True),                   # slider3b
        gr.update(value=1, visible=True),                   # slider3c
        gr.update(value="", visible=True),  # comentario_general
        gr.update(visible=True),  # ranking_instruction
        gr.update(value=None, visible=True),  # ranking1
        gr.update(value=None, visible=True),  # ranking2
        gr.update(value=None, visible=True),  # ranking3
        gr.update(visible=False),                            # reiniciar_btn
        state
    ]

def toggle_slider1(checkbox_value):
    return [gr.update(value=1, interactive=not checkbox_value), gr.update(value=1, interactive=not checkbox_value), gr.update(value=1, interactive=not checkbox_value)]


# Create interface
with gr.Blocks() as demo:
    gr.HTML("""
    <style>
    #ranking-error-msg {
        color: red;
        font-weight: bold;
        margin-bottom: 1em;
    }
    </style>
    """)
    title = gr.Markdown("## Restored Image Comparator")

    # Consent information accordion (shown at the top)
    consentimiento_texto = '''
#INFORMACIÓN AL PARTICIPANTE DEL PROYECTO Recuperar las Memorias.

Este documento tiene por objeto ofrecerle información sobre un proyecto de investigación en el que se le invita a participar. Este estudio ha sido aprobado por el Comité de Ética en Investigación Social de la Universitat Politécnica de Valencia de acuerdo con la legislación vigente, y se lleva a cabo con respeto a los principios enunciados en la declaración del Helsinki y a las normas de buena práctica en investigación.
Si decide autorizar su participación en el mismo, debe recibir información personalizada del investigador, leer antes este documento y hacer todas las preguntas que precise para comprender los detalles sobre este. Este documento puede consultarlo con otras personas y tomarse el tiempo que necesite para decidir si autoriza su participación o no.
La participación en este estudio es completamente voluntaria. Puede decidir no participar o, si acepta hacerlo, cambiar de parecer retirando el consentimiento en cualquier momento sin dar explicaciones.

##¿Cuál es el propósito del estudio?

El objetivo de este proyecto es recabar información sobre la percepción humana sobre la restauración (a través de modelos de Inteligencia Artificial de Apredizaje Profundo) hecha a fotografías dañadas por la DANA.

##¿Por qué me ofrecen participar a mí? ¿Qué tendré que hacer?

Los participantes de este estudio son personas cercanas a los investigadores. 
Para llevar a cabo el estudio tenemos previsto realizar evaluación sobre 3 restauraciones realizadas a cada fotografía dañada de un total de 22 fotografías lo que le implicará un tiempo aproximado de 20 minutos aproximadamente.
Usted se puede beneficiar participando en esta investigación al contribuir a la comparación de percepción humana en contraste con métricas de evaluación de restauración de fotos automática. Por otro lado, cabe destacar que no existen riesgos asociados a su participación.

Por tanto, declaro que mi participación es absolutamente voluntaria y puedo retirarme en cualquier momento sin ningún perjuicio ni penalización.

##¿Qué institución o instituciones participan en el proyecto? ¿Van a obtener beneficios económicos con su desarrollo?

VRAIN y la Escuela de Bellas Artes de la UPV.
Los investigadores no obtendrán beneficios económicos personales directos como producto de la realización de esta encuesta.

##¿Hay alguna restricción de confidencialidad?
Compromiso de Confidencialidad: El participante de esta encuesta se compromete a mantener la estricta confidencialidad sobre las Imágenes expuestas y se obliga a no realizar, directa
ni indirectamente, ninguna de las siguientes acciones:
a. Guardar Copias: El Participante se compromete a no guardar, almacenar ni duplicar
las Imágenes en ningún formato, ya sea físico o digital.
b. Divulgar las Imágenes: El Receptor se compromete a no divulgar, compartir,
distribuir o poner a disposición de terceros las Imágenes, tanto de forma directa como
indirecta.
c. No se permite hacer ningún tipo de obra derivada

##¿Con quién debo contactar si tengo más dudas o no entiendo algo?
Puede contactar con Cèsar Ferri (cferri@dsic.upv.es), Carlos Monserrat (cmonserr@dsic.upv.es), José Daniel Galván (jdgalsua@posgrado.upv.es) y Hugo Albert (halbbon@etsinf.upv.es).
'''
    consent_accordion = gr.Accordion("Información y consentimiento para la participación", open=False)
    with consent_accordion:
        gr.Markdown(consentimiento_texto)

    # Initial screen components
    error_msg = gr.Markdown("", visible=False)
    nombre_input = gr.Textbox(label="Please enter your name", placeholder="Name")
    aceptar_consentimiento_checkbox = gr.Checkbox(label="I accept to participate in this study", value=False)
    comenzar_btn = gr.Button("Start Evaluation", interactive=False)

    # Function to update start button interactivity
    def update_start_button(name, consent_checked):
        return gr.update(interactive=bool(name.strip()) and consent_checked)

    # Link checkbox and name input to button interactivity
    aceptar_consentimiento_checkbox.change(
        fn=update_start_button,
        inputs=[nombre_input, aceptar_consentimiento_checkbox],
        outputs=[comenzar_btn]
    )
    nombre_input.change(
        fn=update_start_button,
        inputs=[nombre_input, aceptar_consentimiento_checkbox],
        outputs=[comenzar_btn]
    )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                progress = gr.Markdown(f"### Original Image 1/{len(imagenes_precargadas)}", visible=False)
            img1 = gr.Image(label="Original", visible=False)
            name1 = gr.Markdown(visible=False)
        # Restoration 1
        with gr.Column(scale=1):
            markdown2 = gr.Markdown("### Restoration 1", visible=False)
            img2 = gr.Image(visible=False)
            name2 = gr.Markdown(visible=False)
            slider1a = gr.Slider(1, 10, step=1, label="Identity preservation", interactive=True, visible=False)
            slider1b = gr.Slider(1, 10, step=1, label="Stain removal", interactive=True, visible=False)
            slider1c = gr.Slider(1, 10, step=1, label="Coherent reconstruction of damaged areas", interactive=True, visible=False)
        # Restoration 2
        with gr.Column(scale=1):
            markdown3 = gr.Markdown("### Restoration 2", visible=False)
            img3 = gr.Image(visible=False)
            name3 = gr.Markdown(visible=False)
            slider2a = gr.Slider(1, 10, step=1, label="Identity preservation", interactive=True, visible=False)
            slider2b = gr.Slider(1, 10, step=1, label="Stain removal", interactive=True, visible=False)
            slider2c = gr.Slider(1, 10, step=1, label="Coherent reconstruction of damaged areas", interactive=True, visible=False)
        # Restoration 3 (SegFormer)
        with gr.Column(scale=1):
            markdown4 = gr.Markdown("### Restoration 3", visible=False)
            img4 = gr.Image(visible=False)
            name4 = gr.Markdown(visible=False)
            slider3a = gr.Slider(1, 10, step=1, label="Identity preservation", interactive=True, visible=False)
            slider3b = gr.Slider(1, 10, step=1, label="Stain removal", interactive=True, visible=False)
            slider3c = gr.Slider(1, 10, step=1, label="Coherent reconstruction of damaged areas", interactive=True, visible=False)
    

    with gr.Row():
        with gr.Column(scale=1):
            markdown1 = gr.Markdown("""
            **How to evaluate the restorations?**  
            - **Identity preservation**: 1 if people are not recognizable, 10 if you can perfectly tell they are the same people. 
            - **Stain removal**: 1 if the stain is the same size or has increased, 10 if the stain has disappeared, regardless of the coherence of the generated parts.
            - **Coherent reconstruction**: 1 if the reconstructed parts have nothing to do with the rest of the image or are not realistic, 10 if the image is coherent and does not look generated, regardless of whether it still has stains.
            """, visible=False)

        with gr.Column(scale=3):
            with gr.Row():
                checkbox1 = gr.Checkbox(label="There are no faces in the original photo", visible=False)
            with gr.Row():
                # Restoration 1
                with gr.Column(scale=1):
                    slider1a = gr.Slider(1, 10, step=1, label="Identity preservation", interactive=True, visible=False)
                    slider1b = gr.Slider(1, 10, step=1, label="Stain removal", interactive=True, visible=False)
                    slider1c = gr.Slider(1, 10, step=1, label="Coherent reconstruction of damaged areas", interactive=True, visible=False)
                # Restoration 2
                with gr.Column(scale=1):
                    slider2a = gr.Slider(1, 10, step=1, label="Identity preservation", interactive=True, visible=False)
                    slider2b = gr.Slider(1, 10, step=1, label="Stain removal", interactive=True, visible=False)
                    slider2c = gr.Slider(1, 10, step=1, label="Coherent reconstruction of damaged areas", interactive=True, visible=False)
                # Restoration 3 (SegFormer)
                with gr.Column(scale=1):
                    slider3a = gr.Slider(1, 10, step=1, label="Identity preservation", interactive=True, visible=False)
                    slider3b = gr.Slider(1, 10, step=1, label="Stain removal", interactive=True, visible=False)
                    slider3c = gr.Slider(1, 10, step=1, label="Coherent reconstruction of damaged areas", interactive=True, visible=False)
        

    gr.HTML("<hr>")

    with gr.Row():
        with gr.Column(scale=2):
            ranking_instruction = gr.Markdown(
                """
                **Rank your restoration preferences**  
                (1 is the one you liked the most and 3 is the one you liked the least)
                """,
                visible=False
            )
            ranking_choices = ["Restoration 1", "Restoration 2", "Restoration 3"]
            ranking1 = gr.Dropdown(
                choices=ranking_choices,
                label="1st preference",
                interactive=True,
                visible=False
            )
            ranking2 = gr.Dropdown(
                choices=ranking_choices,
                label="2nd preference",
                interactive=True,
                visible=False
            )
            ranking3 = gr.Dropdown(
                choices=ranking_choices,
                label="3rd preference",
                interactive=True,
                visible=False
            )
        with gr.Column(scale=2):
            comentario_general = gr.Textbox(lines=15, label="Comments (optional)", visible=False)

    with gr.Row():
        gracias = gr.Markdown("", visible=False)
    
    with gr.Row():
        with gr.Column(scale=2):
            error_msg = gr.Markdown("", visible=False, elem_id="ranking-error-msg")
        with gr.Column(scale=1):
            pass
    with gr.Row():
        btn = gr.Button("Next", visible=False)
        reiniciar_btn = gr.Button("Start New Evaluation", visible=False)

    # Eventos de los checkboxes
    checkbox1.change(fn=toggle_slider1, inputs=[checkbox1], outputs=[slider1a, slider2a, slider3a])

    # Lógica para evitar repeticiones en los dropdowns de ranking
    def update_ranking2(r1):
        if r1:
            return gr.update(choices=[c for c in ranking_choices if c != r1], value=None)
        else:
            return gr.update(choices=ranking_choices, value=None)
    def update_ranking3(r1, r2):
        selected = set([r1, r2])
        return gr.update(choices=[c for c in ranking_choices if c not in selected], value=None)
    ranking1.change(fn=update_ranking2, inputs=[ranking1], outputs=[ranking2])
    ranking2.change(fn=update_ranking3, inputs=[ranking1, ranking2], outputs=[ranking3])

    # Eventos principales: All event handlers are defined after all UI components to ensure components are defined.
    user_state = gr.State(initial_state.copy())
    comenzar_btn.click(
        fn=iniciar_evaluacion,
        inputs=[nombre_input, user_state],
        outputs=[
            error_msg,
            nombre_input,
            comenzar_btn,
            consent_accordion,
            aceptar_consentimiento_checkbox,
            img1, name1,
            img2, name2,
            img3, name3,
            img4, name4,
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
            slider2b,
            slider2c,
            markdown4,
            slider3a,
            slider3b,
            slider3c,
            comentario_general,
            ranking_instruction,
            ranking1, ranking2, ranking3,
            reiniciar_btn,
            user_state
        ]
    )

    btn.click(
        fn=mostrar_siguiente,
        inputs=[
            slider1a, checkbox1, slider1b, slider1c,
            slider2a, slider2b, slider2c,
            slider3a, slider3b, slider3c,
            comentario_general,
            ranking1, ranking2, ranking3,
            user_state
        ],
        outputs=[
            error_msg,
            nombre_input,
            comenzar_btn,
            img1, name1,
            img2, name2,
            img3, name3,
            img4, name4,
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
            slider2b,
            slider2c,
            markdown4,
            slider3a,
            slider3b,
            slider3c,
            comentario_general,
            ranking_instruction,
            ranking1, ranking2, ranking3,
            reiniciar_btn,
            user_state
        ]
    )

    reiniciar_btn.click(
        fn=lambda state: iniciar_evaluacion(state["nombre_usuario"], state),
        inputs=[user_state],
        outputs=[
            error_msg,
            nombre_input,
            comenzar_btn,
            consent_accordion,
            aceptar_consentimiento_checkbox,
            img1, name1,
            img2, name2,
            img3, name3,
            img4, name4,
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
            slider2b,
            slider2c,
            markdown4,
            slider3a,
            slider3b,
            slider3c,
            comentario_general,
            ranking_instruction,
            ranking1, ranking2, ranking3,
            reiniciar_btn,
            user_state
        ]
    )

demo.launch(server_port=7865, debug=True)
