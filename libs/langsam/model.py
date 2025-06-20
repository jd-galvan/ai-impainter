from samgeo.text_sam import LangSAM
from PIL import Image, ImageFilter, ImageOps
import os
import numpy as np
from tqdm import tqdm
import cv2
from skimage.measure import label, regionprops, regionprops_table
from skimage.io import imread, imshow

class LangSAMFaceExtractor():
  def __init__(self, device: str, model_size:["tiny", "small", "base-plus", "large"] = "tiny"):
    self.sam = LangSAM(model_type=f"sam2-hiera-{model_size}")
    self.sam.device = device
    self.model_size = model_size
    print("Modelo LangSAM descargado ✅")

  def print_result(self):
    self.sam.show_anns(
          cmap="Greens",
          box_color="red",
          title="Automatic Segmentation of Faces",
          blend=True,
      )

  def __call__(self, image:[str, Image], box_threshold=0.3,text_threshold=0.24, output:[str, None]=None, mask_multiplier:int=1, dtype=np.uint8, return_results:["mask", "box", "both", None]=None, print_result:bool = True, head:bool = True):
    #Print results only works when result is not returned
    if isinstance(image, str):
      image = Image.open(image)
      image = ImageOps.exif_transpose(image)

    self.sam.predict(image, text_prompt="face", box_threshold=box_threshold, text_threshold=text_threshold, output = output, mask_multiplier=mask_multiplier, dtype=dtype)
    mask = self.sam.prediction.astype(np.uint8)
    boxes = self.sam.boxes.cpu().numpy() #format y1, x1, y2, x2
    if head:
      self.sam.predict(image, text_prompt="head", box_threshold=box_threshold, text_threshold=text_threshold, output = output, mask_multiplier=mask_multiplier, dtype=dtype)
      mask = np.clip(mask + self.sam.prediction.astype(np.uint8), 0, mask_multiplier)
      grayImage = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
      kernel = np.ones((30, 30), np.uint8)
      closing = cv2.morphologyEx(grayImage, cv2.MORPH_CLOSE, kernel)
      closing = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)
      mask = closing
      label_im = label((closing>0).astype(int))
      regions = regionprops(label_im)
      boxes = np.array([np.array([region.bbox[1], region.bbox[0], region.bbox[3], region.bbox[2]]) for region in regions])


    if return_results == "mask": return mask
    elif return_results == "box": return boxes
    elif return_results == "both": return mask, boxes

    if print_result:
      self.print_result()
