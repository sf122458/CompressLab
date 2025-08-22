import logging
import torch
from inspect import isfunction

def autocast(f):
    def do_autocast(*args, **kwargs):
        with torch.amp.autocast(
            "cuda", 
            enabled=True,
            dtype=torch.get_autocast_gpu_dtype(),
            cache_enabled=torch.is_autocast_cache_enabled()
        ):
            return f(*args, **kwargs)

    return do_autocast


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        logging.info(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params