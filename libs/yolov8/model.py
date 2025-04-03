import os
import cv2
from PIL import Image
from ultralytics import YOLO

class YOLOV8:
    def __init__(self, model_path: str=None):
      self.model = None
      if model_path is not None:
        self.model = YOLO(model_path) 

    def set_model(self, model_path):
      self.model = YOLO(model_path)
    
    def get_bounding_box(self, confidence: float, image_path: str):
      results = self.model.predict(source=image_path, save=False, conf=confidence, verbose=False)
    
      # Imagen con bounding boxes dibujadas
      annotated_frame = results[0].plot()
      annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
      image_with_boxes = Image.fromarray(annotated_frame_rgb)

      # Obtener bounding boxes
      boxes = results[0].boxes
      coordinates = boxes.xyxy.cpu().numpy()     # (x1, y1, x2, y2)
      bbox_coords = [list(map(float, coord)) for coord in coordinates]

      return image_with_boxes, bbox_coords
