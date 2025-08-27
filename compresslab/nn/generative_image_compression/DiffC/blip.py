from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
from pathlib import Path
import pandas as pd
from compresslab.utils.constant import PRETRAINED_CACHE_DIR
from lightning import LightningModule

class BlipCaptioner(LightningModule):
    def __init__(
        self, 
        # model_name: str = "Salesforce/blip2-opt-2.7b-coco",  # NOTE: this ckpt will cause error
        model_name: str = "Salesforce/blip2-opt-2.7b",
        max_length: int = 75,
        cache_dir: str = PRETRAINED_CACHE_DIR,
    ):
        """Initialize BLIP-2 model for batch captioning."""
        super().__init__()
        self.processor = Blip2Processor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)
        self.max_length = max_length
        
    def generate_caption(self, image: torch.Tensor) -> str:
        """Generate caption for a single image."""
        inputs = self.processor(images=image, return_tensors="pt", do_rescale=False).to(self.device)
        generated_ids = self.model.generate(**inputs, max_length=self.max_length)
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return caption

    def __del__(self):
        """Cleanup any GPU memory."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
