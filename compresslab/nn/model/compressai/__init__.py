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

from compresslab.config import Config
from compresslab.utils.base import  _baseCompound
from compresslab.loss import LossFn
from compresslab.utils.registry import ModelRegistry, CompoundRegistry
from compressai.models import *

@CompoundRegistry.register("CompressAI")
class Compound(_baseCompound):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config=config, **kwargs)
        assert issubclass(self.model.__class__, CompressionModel), "Model should be a subclass of `CompressModel`."
        self.bpp_weight = kwargs["loss_config"]["bpp"]["weight"]
    
    def __call__(self, x: torch.Tensor):
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
        
        loss = distortion_loss + bpp_loss * self.bpp_weight

        aux_loss = self.model.aux_loss()

        return \
            {   
                "loss": loss,
                "log":
                {
                    "psnr": 10 * torch.log10(1 / torch.nn.functional.mse_loss(out["x_hat"], x)).item(),
                    "bpp": bpp_loss,
                    "total_loss": loss,
                },
                "x_hat": out["x_hat"],
                "likelihoods": out["likelihoods"],
                "bpp_loss": bpp_loss,
                "aux_loss": aux_loss
            }
    
    def compress(self, x: torch.Tensor) -> Dict[str, Any]:
        """
            return: {"strings": [y_strings], "shape": y.size()[-2:]} | {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
        """
        return self.model.compress(x)
    
    def decompress(self, string: bytes, shape: torch.Size) -> Dict[str, torch.Tensor]:
        """
            return: {"x_hat": x_hat}
        """
        return self.model.decompress(string, shape)




ModelRegistry.register("FactorizedPrior")(FactorizedPrior)

ModelRegistry.register("ScaleHyperprior")(ScaleHyperprior)

ModelRegistry.register("MeanScaleHyperprior")(MeanScaleHyperprior)

ModelRegistry.register("JointAutoregressiveHierarchicalPriors")(JointAutoregressiveHierarchicalPriors)

ModelRegistry.register("Cheng2020Attention")(Cheng2020Attention)

ModelRegistry.register("Cheng2020AnchorCheckerboard")(Cheng2020AnchorCheckerboard)

ModelRegistry.register("Cheng2020Anchor")(Cheng2020Anchor)

ModelRegistry.register("Elic2022Official")(Elic2022Official)

ModelRegistry.register("Elic2022Chandelier")(Elic2022Chandelier)