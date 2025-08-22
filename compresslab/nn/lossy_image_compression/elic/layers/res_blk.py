import torch.nn as nn
from compresslab.core.layers import conv1x1


class ResidualBottleneck(nn.Module):
    def __init__(self, N=192, act=nn.ReLU) -> None:
        super().__init__()
        self.branch = nn.Sequential(
            conv1x1(N, N // 2),
            act(),
            nn.Conv2d(N // 2, N // 2, kernel_size=3, stride=1, padding=1),
            act(),
            conv1x1(N // 2, N)
        )

    def forward(self, x):
        out = x + self.branch(x)

        return out