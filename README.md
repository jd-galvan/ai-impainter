# 🧼 Stain Remover

## ✨ Descripción

`stain-remover` es una aplicación en Python diseñada para eliminar manchas en fotografías utilizando modelos avanzados de inteligencia artificial y procesamiento de imágenes. Este proyecto ha sido desarrollado en la **Universidad Politécnica de Valencia (UPV)** como parte del proyecto **Salvem Les Fotos**.

Hace uso de las siguientes tecnologías:

- 🖼️ **SAM2** (Segment Anything Model v2) para la segmentación de manchas.
- 🎨 **Stable Diffusion Inpainting XL** para la restauración de imágenes.
- 🏞️ **OpenCV** para el procesamiento de imágenes.
- 🔍 **BLIP** (Bootstrapped Language-Image Pretraining) para mejorar la interpretación de la imagen.
- 🌐 **Gradio** para la creación de una interfaz web accesible.

## ⚙️ Requisitos

- 🐍 Python >= 3.10
- 🚀 CUDA-compatible GPU (opcional, pero recomendado para un mejor rendimiento)

## 📥 Instalación

### 1️⃣ Clonar el repositorio

```bash
 git clone https://github.com/tuusuario/stain-remover.git
 cd stain-remover
```

### 2️⃣ Crear y activar un entorno virtual (opcional pero recomendado)

```bash
python -m venv venv
source venv/bin/activate  # En Linux/macOS
venv\Scripts\activate  # En Windows
```

### 3️⃣ Instalar las dependencias

```bash
pip install -r requirements.txt
```

## 🛠️ Configuración de Variables de Entorno

Este proyecto requiere la configuración de variables de entorno para su correcto funcionamiento. Se proporciona un archivo `.env-example` como referencia.

### 📌 Pasos:

1. Copia el archivo `.env-example` y renómbralo como `.env`:
   ```bash
   cp .env-example .env
   ```
2. Edita el archivo `.env` y completa los valores de las siguientes variables:
   ```env
   CUDA_DEVICE=cuda:0  # Puedes configurar "cuda:0", "cuda:1" o la tarjeta gráfica que desees usar.
   HUGGINGFACE_HUB_TOKEN=tu_token_aquí
   ```

## 🚀 Uso

Para ejecutar la aplicación, simplemente corre el siguiente comando:

```bash
python app.py
```

Esto iniciará una interfaz web con **Gradio** donde podrás cargar imágenes y procesarlas para eliminar manchas.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Si deseas mejorar el proyecto, por favor:

1. Haz un fork del repositorio.
2. Crea una nueva rama (`git checkout -b mi-nueva-caracteristica`).
3. Realiza tus cambios y confirma los commits.
4. Envía un pull request.

## 📜 Licencia

Este proyecto está bajo la licencia MIT. Para más detalles, consulta el archivo `LICENSE`. 🚀

