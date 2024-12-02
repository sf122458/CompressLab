from compresslab.utils.registry import LossRegistry
from compresslab.config import Config
import torch
import torch.nn as nn
import logging

class LossFn(nn.Module):
    def __init__(self, config: Config):
        super(LossFn, self).__init__()
        self.loss = []
        self.lmbda = []
        self.config = config
        logging.debug(config.Train.Loss)
        for type, lmbda in config.Train.Loss.items():
            try:
                loss_fn = LossRegistry.get(type)()
                self.loss.append(loss_fn)
                self.lmbda.append(lmbda)
            except:
                if type.upper() == "BPP":
                    logging.info(f"Find bpp loss. Please check the compound has implemented the bpp loss calculation.")
                else:
                    logging.warning(f"Loss function {type} is not implemented. Skip this loss function.")
            

    def forward(self, x, xHat, **kwargs):
        loss = 0
        for l, lmbda in zip(self.loss, self.lmbda):
            loss += l(x, xHat) * lmbda
        return loss


@LossRegistry.register("MSE")
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, x, y):
        return torch.mean((x - y) ** 2)
    
@LossRegistry.register("L1")
class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, x, y):
        return torch.mean(torch.abs(x - y))
    
@LossRegistry.register("SSIM")
class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, x, y):
        return 1 - torch.mean(torch.nn.functional.ssim(x, y))
    
@LossRegistry.register("MSSSIM")
class MSSSIMLoss(nn.Module):
    def __init__(self):
        super(MSSSIMLoss, self).__init__()

    def forward(self, x, y):
        return 1 - torch.mean(torch.nn.functional.msssim(x, y))
    
@LossRegistry.register("SmoothL1")
class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()

    def forward(self, x, y):
        return torch.mean(torch.nn.functional.smooth_l1_loss(x, y))
    
@LossRegistry.register("CrossEntropy")
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, x, y):
        return torch.nn.functional.cross_entropy(x, y)