from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import sys
sys.path.append("..")


class SAM2:
    def __init__(self, device: str):
        sam_checkpoint = "sam_vit_h_4b8939.pth"
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
