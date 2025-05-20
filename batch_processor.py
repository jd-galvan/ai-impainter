import os
import glob
import shutil
import time
import threading
import cv2
import gradio as gr
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from libs.sam2.model import SAM2
from libs.stable_diffusion.impaint.model import SDImpainting
from libs.yolov8.model import YOLOV8
from utils import (
    generate_binary_mask,
    delete_irrelevant_detected_pixels,
    fill_little_spaces,
    soften_contours
)

load_dotenv()

# Configuración del dispositivo para modelos
DEVICE = os.environ.get("CUDA_DEVICE")
print(f"DEVICE {DEVICE}")

# Cargar modelos
segmentation_model = SAM2(DEVICE)
impainting_model = SDImpainting(DEVICE)
yolo_model = YOLOV8(device=DEVICE)
paths = glob.glob("./tools/trainer/yolov8/runs/detect/*/weights/best.pt")
yolo_model.set_model(paths[0])


# ====== Estado Global Compartido y Mecanismo de Bloqueo ======
shared_processing_data = []
processing_lock = threading.Lock()
is_processing = False
# ===========================================================

# Obtener la ruta del directorio home del usuario
home_directory = os.path.expanduser('~')


# Función para obtener el estado actual del procesamiento compartido (sin controlar el estado del botón)
def get_current_processing_state():
    """
    Retorna el estado actual de los datos de procesamiento compartidos y un mensaje de estado general.
    """
    global shared_processing_data, is_processing
    message = "Selecciona archivos para iniciar el proceso."
    if is_processing:
        message = "⏳ Ya hay un proceso de restauración en curso. Verificando estado..."
    elif shared_processing_data:  # Si hay datos pero no está procesando, mostramos el último estado completado/con error
        message = "🎉 Último proceso completado."

    # Devolvemos los datos de la tabla y el mensaje de estado
    return shared_processing_data, message


# Función que maneja el clic del botón, inicia el procesamiento y actualiza el estado (sin deshabilitar el botón)
def handle_processing_click(lista_elementos_seleccionados):
    global shared_processing_data, is_processing

    # Si ya hay un proceso en curso, notificamos y devolvemos el estado actual
    if is_processing:
        # El estado del botón no se controla aquí directamente
        return shared_processing_data, "⏳ Ya hay un proceso de restauración en curso. Por favor espera a que termine."

    # Si no hay proceso en curso, intentamos adquirir el bloqueo
    with processing_lock:
        # Verificación doble dentro del bloqueo
        if is_processing:
            return shared_processing_data, "⏳ Ya hay un proceso de restauración en curso. Por favor espera a que termine."

        is_processing = True  # Marcamos que el proceso está en curso
        shared_processing_data = []  # Limpiamos datos anteriores

        rutas_archivos = []
        if lista_elementos_seleccionados:
            # Ignoramos el primer elemento asumiendo que es la carpeta
            rutas_archivos = lista_elementos_seleccionados[1:]

        # Si no hay archivos seleccionados, salimos
        if not rutas_archivos:
            is_processing = False  # Resetear bandera
            return [], "⚠️ No hay archivos seleccionados para procesar."

        # 1. Preparar datos iniciales de la tabla con estado 'Pendiente'
        for ruta in rutas_archivos:
            shared_processing_data.append([ruta, "✨ Pendiente", ""])

        # Generar estado inicial: tabla y mensaje
        yield shared_processing_data, "✅ Proceso de restauración iniciado. Procesando archivos..."

        # 2. Procesar cada archivo
        for i, ruta_original in enumerate(rutas_archivos):
            shared_processing_data[i][1] = "⏳ Procesando..."
            begin = time.time()
            # Generar actualizaciones: tabla y mensaje (sin controlar el botón)
            yield shared_processing_data, f"⏳ Procesando archivo {i+1}/{len(rutas_archivos)}..."

            try:
                print("YOLO detection started 🔍")
                yolo_image, boxes = yolo_model.get_bounding_box(
                    0.1, ruta_original)
                print(
                    f"YOLO detection has finished succesfully. {len(boxes)} boxes")

                image = cv2.imread(ruta_original)
                if image is None:
                    raise ValueError(
                        "No se pudo cargar la imagen. Verifica la ruta.")

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                binary_mask = None
                # Generación de la máscara
                print(f"SAM detection for {len(boxes)} box started 🔬")
                masks = segmentation_model.get_mask_by_bounding_boxes(
                    boxes=boxes, image=image)
                print(f"SAM detection has finished successfully")

                # Mezclar multiples mascaras de SAM
                numpy_masks = [mask.cpu().numpy() for mask in masks]
                combined_mask = np.zeros_like(numpy_masks[0], dtype=bool)
                for m in numpy_masks:
                    combined_mask = np.logical_or(combined_mask, m)

                # Generar mascara binaria
                binary_mask = generate_binary_mask(combined_mask)

                # Guardar mascara original
                directorio, nombre_completo = os.path.split(ruta_original)
                nombre, extension = os.path.splitext(nombre_completo)
                ruta_mascara_original = os.path.join(
                    directorio, f"{nombre}_MASK_ORIGINAL.png")
                processed_mask.save(ruta_mascara_original)

                print("Refining generated mask with OpenCV 🖌")
                refined_binary_mask = delete_irrelevant_detected_pixels(
                    binary_mask)
                without_irrelevant_pixels_mask = fill_little_spaces(
                    refined_binary_mask)
                dilated_mask = soften_contours(
                    without_irrelevant_pixels_mask)
                blurred_mask = dilated_mask
                print("Image was refined successfully!")

                # Guardar máscara refinada
                processed_mask = Image.fromarray(blurred_mask, mode='L')
                ruta_mascara_final = os.path.join(
                    directorio, f"{nombre}_MASK_REFINED.png")
                processed_mask.save(ruta_mascara_final)

                print("SD XL Impainting started 🎨")
                new_image = impainting_model.impaint(
                    image_path=ruta_original,
                    mask_path=ruta_mascara_final,
                    prompt="photo restoration, realistic, same style",
                    strength=0.99,
                    guidance=7,
                    padding_mask_crop=None,
                    steps=20,
                    negative_prompt="blurry, distorted, unnatural colors, artifacts, harsh edges, unrealistic texture, visible brush strokes, AI look, text",
                    keep_faces=True,
                    see_face_masks=False
                )
                print("SD XL Impainting process finished")

                ruta_restauracion = os.path.join(
                    directorio, f"{nombre}_RESTORED.png")
                new_image.save(ruta_restauracion)

                end = time.time()
                duration = round(end-begin, 3)
                shared_processing_data[i][1] = "✅ Restaurado"
                shared_processing_data[i][2] = f"{duration}"
            except Exception as e:
                # Actualizar estado en caso de error
                if len(shared_processing_data[i]) > 1:
                    shared_processing_data[i][1] = f"❌ Error: {e}"
                    shared_processing_data[i][2] = "-"

            # Generar actualizaciones: tabla y mensaje (sin controlar el botón)
            yield shared_processing_data, f"✅ Archivo {i+1}/{len(rutas_archivos)} procesado."

        # Después de que el bucle termine
        is_processing = False  # Proceso terminado

        # Último estado: tabla y mensaje final
        yield shared_processing_data, "🎉 Proceso de restauración completado."


# Creamos la interfaz de Gradio
# Añadimos un título para la pestaña del navegador
with gr.Blocks(title="AI-Impainter: Restauración de Fotos de la DANA") as demo:
    gr.Markdown(
        """
        # 🎨 Salvem les fotos - Restauración de Fotos con IA 📸

        Bienvenido a **AI-Impainter**, tu herramienta especializada en devolver la vida a esas preciadas fotos
        afectadas por el barro y el agua de la DANA. ✨

        Selecciona las imágenes que deseas restaurar utilizando el explorador de archivos y
        presiona el botón para iniciar el proceso de restauración con nuestra inteligencia artificial. 🖌️
        """
    )

    # Componente FileExplorer para seleccionar archivos/carpetas.
    file_explorer = gr.FileExplorer(
        root_dir=home_directory,
        file_count="multiple",
        label=f"📂 Selecciona las Imágenes a Restaurar (Inicio: {home_directory})"
    )

    # Componente Textbox para mostrar mensajes de estado general del proceso.
    # Se actualizará al cargar la página y al presionar el botón.
    status_message = gr.Textbox(
        label="Estado General del Proceso",
        interactive=False,
    )

    # Definimos el botón de procesamiento
    procesar_button = gr.Button(
        "✨ Iniciar Restauración con IA ✨")  # Botón con emojis

    # Componente Dataframe para mostrar la lista de rutas de archivos y su estado.
    # Se actualizará al cargar la página y al presionar el botón.
    output_tabla_procesamiento = gr.Dataframe(
        label="📊 Estado del Procesamiento de Archivos",  # Etiqueta con emoji
        headers=["Ruta del Archivo", "Estado",
                 "Tiempo (s)"],  # Cabeceras de la tabla
        interactive=False,  # La tabla no es editable por el usuario
    )

    # ====== Manejo del Estado Compartido y Concurrencia ======

    # Carga el estado actual al cargar la página.
    demo.load(
        fn=get_current_processing_state,  # Llama a la función que obtiene el estado global
        outputs=[output_tabla_procesamiento, status_message],
        queue=False  # Es importante que los eventos load no se encolen
    )

    # Conecta el botón al manejador del proceso.
    # Actualiza la tabla y el mensaje de estado (sin controlar el estado del botón).
    procesar_button.click(
        fn=handle_processing_click,  # Llama a la función que maneja el clic y el proceso
        inputs=file_explorer,  # La entrada es el valor actual del file_explorer
        # Las salidas son la tabla y el mensaje de estado
        outputs=[output_tabla_procesamiento, status_message]
        # Gradio gestionará automáticamente el encolamiento si múltiples usuarios presionan el botón
    )

# Lanzamos la interfaz de Gradio.
demo.launch(debug=True, auth=(os.environ.get(
    "APP_USER"), os.environ.get("APP_PASSWORD")))
