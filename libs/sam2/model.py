import os
import urllib.request
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import sys
sys.path.append("..")


class SAM2:
    def __init__(self, device: str):
        self.checkpoints_dir = "./checkpoints/"
        self.model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        self.model_path = os.path.join(
            self.checkpoints_dir, os.path.basename(self.model_url))

        # Crear el directorio si no existe
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Descargar el archivo si no existe
        if not os.path.exists(self.model_path):
            print(f"Descargando el modelo desde {self.model_url}...")
            urllib.request.urlretrieve(self.model_url, self.model_path)
            print("Descarga completa.")
        else:
            print("El modelo ya está descargado.")

        sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
        model_type = "vit_h"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.predictor = SamPredictor(sam)

    def get_mask_by_pixel(self, x: float, y: float, image: np.ndarray):
        # Pixel seleccionado
        input_point = np.array([[x, y]])
        # Etiqueta para el punto seleccionado
        input_label = np.array([1])

        # Configurar la imagen en el predictor
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        return masks[0]
