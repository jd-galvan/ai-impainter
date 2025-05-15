import gradio as gr
import os
import shutil
import time
import threading
from dotenv import load_dotenv

load_dotenv()

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
            shared_processing_data.append([ruta, "✨ Pendiente"])

        # Generar estado inicial: tabla y mensaje
        yield shared_processing_data, "✅ Proceso de restauración iniciado. Procesando archivos..."

        # 2. Procesar cada archivo
        for i, ruta_original in enumerate(rutas_archivos):
            shared_processing_data[i][1] = "⏳ Procesando..."
            # Generar actualizaciones: tabla y mensaje (sin controlar el botón)
            yield shared_processing_data, f"⏳ Procesando archivo {i+1}/{len(rutas_archivos)}..."

            try:
                directorio, nombre_completo = os.path.split(ruta_original)
                nombre, extension = os.path.splitext(nombre_completo)
                nueva_ruta = os.path.join(
                    directorio, f"{nombre}_AI{extension}")

                shutil.copy2(ruta_original, nueva_ruta)
                time.sleep(5)

                shared_processing_data[i][1] = "✅ Restaurado"
            except Exception as e:
                # Actualizar estado en caso de error
                if len(shared_processing_data[i]) > 1:
                    shared_processing_data[i][1] = f"❌ Error: {e}"

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
        headers=["Ruta del Archivo", "Estado"],  # Cabeceras de la tabla
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
