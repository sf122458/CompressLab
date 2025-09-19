import torch
from torch import nn
from torch import Tensor

def unsqueeze_right(x: Tensor, num_dims: int = 1) -> Tensor:
    """Unsqueezes the last `num_dims` dimensions of `x`."""
    return x.view(x.shape + (1,) * num_dims)

@torch.no_grad()
def zero_init(module: nn.Module) -> nn.Module:
    """Sets to zero all the parameters of a module, and returns the module."""
    for p in module.parameters():
        nn.init.zeros_(p.data)
    return module

def kl_std_normal(mean_squared: Tensor, var: Tensor) -> Tensor:
    return 0.5 * (var + mean_squared - torch.log(var.clamp(min=1e-15)) - 1.0)