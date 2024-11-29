
class _baseTrainer:
    def __init__(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
    
    def val(self):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError
    
"""
    Compound is responsible for the feedforward, compression and decompression process of the model and the loss calculation.
    Components of the Compound class should include the backbone model, the loss function. These should be implemented in `__init__` method.
    In the `forward` method, the Compound class should return a dictionary with specific keys.
"""

import torch.nn as nn
import torch
from compressai.models import CompressionModel

class _baseCompound(nn.Module):
    def __init__(self):
        super().__init__()

        self.model: CompressionModel = None
        self.loss = None
    
    def train(self, mode: bool=True):
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor):
        """
        Need to implement:
            Return should be a dictionary with the following keys
                - loss: total loss value
                - log: benchmark required to be recorded, e.g. PSNR, SSIM, etc.
            and other keys according to the task, such as x_hat, likelihoods, etc in image compression.
        """
        raise NotImplementedError
    
    # def inference(self, x: torch.Tensor):
    #     raise NotImplementedError
    
    def compress(self, x: torch.Tensor):
        raise NotImplementedError
    
    def decompress(self, x: torch.Tensor):
        raise NotImplementedError