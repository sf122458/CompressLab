import torch
import torch.nn as nn
from transformers import Dinov2WithRegistersModel, DINOv3ViTModel
from torchvision import transforms
from compresslab.utils.constant import PRETRAINED_CACHE_DIR
from .augument import DiffAugment


class DINOV2(nn.Module):
    def __init__(self, diff_aug: bool = True):
        super().__init__()
        self.diff_aug = diff_aug
        
        self.backbone = Dinov2WithRegistersModel.from_pretrained(
            "facebook/dinov2-with-registers-base", 
            cache_dir=PRETRAINED_CACHE_DIR
        )
        self.backbone.eval()
        self.backbone.requires_grad_(False)
        
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
            ]
        )

        self.policy = 'translation,cutout'
    
    def to(self, device):
        self.backbone.to(device)
        return self
        
    def dinov2_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x, output_hidden_states=True)
        unused_token_num = 5  # 1 CLS + 4 register tokens
        image_features = x.last_hidden_state[:, unused_token_num:]
        return image_features

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transform(x)
        x_aug = DiffAugment(x, policy=self.policy) if self.diff_aug else x
        return self.dinov2_forward(x_aug)
    
    def state_dict():
        return {}
    
    
class DINOV3:
    pass