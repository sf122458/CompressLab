import torch
import torch.nn as nn
from cbench.utils.registry import ModelRegistry


@ModelRegistry.register("Balle")
class Balle2017(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x