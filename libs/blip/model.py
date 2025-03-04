import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class BLIP:
    def __init__(self, device: str):
        self.device = device
        
        huggingface_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base", token=huggingface_token)
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", token=huggingface_token).to(self.device)

    def generate_caption(self, image_path: str):
        """Detect what is in the picture"""
        raw_image = Image.open(image_path).convert('RGB')
        inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)
