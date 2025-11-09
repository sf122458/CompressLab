import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection
from torchvision import transforms
from typing import Dict, Any
from lightning import LightningModule
from compresslab.utils.constant import PRETRAINED_CACHE_DIR

class FeatureLoss(LightningModule):
    """FeatureLoss
    
    Calculate the similarity of the embeddings from the pretrained vit models.
    """
    def __init__(
        self,
        backbone: str = "clip",
        loss_type: str = "cos",
    ):
        super().__init__()
        
        assert backbone in ["clip"], f"Backbone {backbone} not supported."
        assert loss_type in ["l2", "cos"], f"Loss type {loss_type} not supported."
        
        # use a dict to store models and avoid saving them in checkpoints
        self.models: Dict[str, nn.Module] = dict()
        self.transform: Dict[str, Any] = dict()
        
        self.backbone = backbone.lower()
        self.loss_type = loss_type.lower()
        
        if backbone == "clip":
            # NOTE: assume input is in [0, 1] tensor format
            self.transform["clip"] = transforms.Compose([
                transforms.Resize(224),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073], 
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
            self.models["clip"] = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-base-patch32",
                cache_dir=PRETRAINED_CACHE_DIR
            ).eval()
            
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented.")
        
        # Freeze model parameters
        for model in self.models.values():
            model.requires_grad_(False)
            
        self.eval()
         
    def __call__(self, x, x_hat):
        x_transform = self.transform[self.backbone](x)
        x_hat_transform = self.transform[self.backbone](x_hat)
        
        # extract features from pretrained backbone
        features_x = self.models[self.backbone](x_transform).image_embeds
        features_x_hat = self.models[self.backbone](x_hat_transform).image_embeds
        
        # l2 normalize
        features_x = features_x / features_x.norm(p=2, dim=-1, keepdim=True)
        features_x_hat = features_x_hat / features_x_hat.norm(p=2, dim=-1, keepdim=True)
        
        if self.loss_type == "l2":
            loss = nn.functional.mse_loss(features_x, features_x_hat, reduction="mean")
        elif self.loss_type == "cos":
            loss = 1 - nn.functional.cosine_similarity(features_x, features_x_hat, dim=1).mean()
            
        return loss
            
    def on_train_start(self):
        for model in self.models.values():
            model.to(self.device)
            
    def on_validation_start(self):
        for model in self.models.values():
            model.to(self.device)
            
    def on_test_start(self):
        for model in self.models.values():
            model.to(self.device)