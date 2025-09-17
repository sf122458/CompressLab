import torch
from torch import nn, Tensor

class FixedLinearSchedule(nn.Module):
    def __init__(self, gamma_min: float, gamma_max: float):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def forward(self, t: Tensor) -> Tensor:
        return self.gamma_min + (self.gamma_max - self.gamma_min) * t


class LearnedLinearSchedule(nn.Module):
    def __init__(self, gamma_min: float, gamma_max: float):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(gamma_min))
        self.w = nn.Parameter(torch.tensor(gamma_max - gamma_min))

    def forward(self, t: Tensor) -> Tensor:
        return self.b + self.w.abs() * t