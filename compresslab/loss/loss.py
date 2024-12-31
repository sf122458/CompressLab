from compresslab.utils.registry import LossRegistry
from compresslab.config import Config
import torch
import torch.nn as nn
import logging
from pytorch_msssim import ssim, ms_ssim

class _baseLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def _forward(self, x, y):
        raise NotImplementedError

    def forward(self, x, y):
        return self._forward(x, y) * self.weight

@LossRegistry.register("MSE")
class MSELoss(_baseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x, y):
        return torch.nn.functional.mse_loss(x, y)
    
@LossRegistry.register("L1")
class L1Loss(_baseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x, y):
        return torch.nn.functional.l1_loss(x, y)
    
@LossRegistry.register("SSIM")
class SSIMLoss(_baseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x, y):
        return 1 - torch.mean(ssim(x, y))
    
@LossRegistry.register("MSSSIM")
class MSSSIMLoss(_baseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x, y):
        return 1 - torch.mean(ms_ssim(x, y))
    
@LossRegistry.register("SmoothL1")
class SmoothL1Loss(_baseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x, y):
        return torch.nn.functional.smooth_l1_loss(x, y)
    
@LossRegistry.register("CrossEntropy")
class CrossEntropyLoss(_baseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x, y):
        return torch.nn.functional.cross_entropy(x, y)