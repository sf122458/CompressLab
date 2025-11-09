import torch.nn as nn
import torch
from torch.nn.utils import spectral_norm
from .backbone import *
from .loss import sigmoid_loss, hinge_loss
from lightning import LightningModule

class MLPD(nn.Module):
    def __init__(self, in_ch=768, out_ch=256, activation=nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()
        print(activation)
        self.decoder = nn.Sequential(
            spectral_norm(nn.Linear(in_ch, out_ch)),
            activation,
        )
        self.out = spectral_norm(nn.Linear(out_ch, 1))

    def forward(self, x):
        h = self.decoder(x)
        out = self.out(h)
        return out


class Discriminator(LightningModule):
    def __init__(
        self, 
        backbone: str = "dinov2",
        loss_type: str = "sigmoid",
        diff_aug: bool = True,
        activation=nn.LeakyReLU(0.2, inplace=True)
    ):
        super().__init__()
        
        assert backbone in ["dinov2"], "Only dinov2 backbone is supported currently."
        assert loss_type in ["sigmoid", "hinge"], "Only sigmoid and hinge loss are supported currently."
        
        if backbone == "dinov2":
            self.backbone = DINOV2(diff_aug=diff_aug)
            
        if loss_type == "sigmoid":
            self.loss_fn = sigmoid_loss()
        else:
            self.loss_fn = hinge_loss()
        
        self.decoder = MLPD(in_ch=768, out_ch=256, activation=activation)
    
    def generator_loss(self, x):
        # Freeze the discriminator parameters
        for p in self.decoder.parameters():
            p.requires_grad = False
    
        feat = self.backbone(x)
        return self.loss_fn(self.decoder(feat), for_real=True)

    
    def discriminator_loss(self, real, fake):
        # Unfreeze the discriminator parameters
        for p in self.decoder.parameters():
            p.requires_grad = True
            
        real_feat = self.backbone(real)
        fake_feat = self.backbone(fake)
            
        real_loss = self.loss_fn(self.decoder(real_feat), for_real=True)
        fake_loss = self.loss_fn(self.decoder(fake_feat), for_real=False)
        
        return (real_loss + fake_loss) / 2
    
    def wrap_optimizer(self, lr=1e-4):
        return torch.optim.AdamW(
            self.decoder.parameters(),
            lr=lr,
        )
        
    def on_train_start(self):
        self.backbone.to(self.device)
        
    def on_validation_start(self):
        self.backbone.to(self.device)
        
    def on_test_start(self):
        self.backbone.to(self.device)