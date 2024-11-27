"""
Model: 
    the architecture of the model used in compression, support forward, inference, compress, decompress

Compound: 
    interact with the model and the loss function(CompressAI models needs to consider the bpp loss and aux loss),
    return the final loss and the output of the model, which will be further processed in trainer
"""


import math
import torch
import torch.nn as nn
from typing import Dict, List, Any

from cbench.config.config import Config
from cbench.nn.model.base import  _baseCompound
from cbench.utils.registry import ModelRegistry, CompoundRegistry, LossRegistry
# from compressai
# from compressai.models import *

@CompoundRegistry.register("CompressAI")
class Compound(_baseCompound):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.model = ModelRegistry.get(config.Model.Net.Key)(**config.Model.Net.Params)
        self.loss = LossRegistry.get(config.Train.Loss)()


    def train(self, mode: bool=True):
        self.model.train(mode)
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor):
        """
        In CompressAI implementation, the output follows the format:
            {
                "x_hat": x_hat,
                "likelihoods": {
                    "y": y_likelihoods,
                    "z": z_likelihoods,
                },
            }
        """
        N, _, H, W = x.size()
        out = self.model(x)

        distortion_loss = self.loss(out["x_hat"], x)
        bpp_loss = \
            sum(
                torch.log(likelihoods).sum() / (-math.log(2) * N * H * W)
                for likelihoods in out["likelihoods"].values()
            )
        
        loss = distortion_loss + bpp_loss * self.config.Train.Loss["bpp"]

        return \
            {   
                "loss": loss,
                "log":
                {
                    "psnr": 10 * torch.log10(1 / distortion_loss).item(),
                },
                "x_hat": out["x_hat"],
                "likelihoods": out["likelihoods"],
                "distortion_loss": distortion_loss,
                "bpp_loss": bpp_loss,
            }
    
    def compress(self, x: torch.Tensor) -> Dict[str, Any]:
        return self.model.compress(x)
    
    def decompress(self, string: bytes, shape) -> Dict[str, torch.Tensor]:
        return self.model.decompress(string, shape)




@ModelRegistry.register("Balle")
class Balle2017(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x