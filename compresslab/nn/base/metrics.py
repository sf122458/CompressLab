import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Any, Union
from dataclasses import dataclass, field
from pytorch_msssim import ms_ssim as ms_ssim_func
# TODO: use torchmetrics if possible
from torchmetrics.functional.image import (
    deep_image_structure_and_texture_similarity as dists_func,
    learned_perceptual_image_patch_similarity as lpips_func,
)
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

@dataclass
class MetricsConfig:
    VAE_IMAGE_COMPRESSION = {
        "train": ["MSE", "MS_SSIM"],
        "eval": ["MSE", "MS_SSIM"]
    }
    
    GENERATIVE_IMAGE_COMPRESSION = {
        "train": ["MSE", "MS_SSIM", "LPIPS"],
        "eval": ["MSE", "MS_SSIM", "KID", "FID", "LPIPS", "DISTS"]
    }

@dataclass
class ImageMetricsOutput:
    bpp: Union[float, torch.Tensor]
    mse_loss: torch.Tensor = field(default=None)
    psnr: float = field(default=None)
    ms_ssim_loss: torch.Tensor = field(default=None)
    ms_ssim: float = field(default=None)
    kid: torch.Tensor = field(default=None)
    fid: torch.Tensor = field(default=None)
    lpips: torch.Tensor = field(default=None)
    dists: torch.Tensor = field(default=None)
        

class MetricsCollector(nn.Module):
    def __init__(
        self, 
        config: Dict[str, Any] = MetricsConfig.VAE_IMAGE_COMPRESSION,
    ):
        super().__init__()
        self.config = config
        
        # self.kid_metric = KernelInceptionDistance(normalize=True)
        # self.fid_metric = FrechetInceptionDistance(normalize=True)
        
        # self._modules.pop('kid_metric', None)
        # self._modules.pop('fid_metric', None)
        
        # self.kid_metric
    
    def reset_config(self, config: Dict[str, Any]):
        """
        Reset the configuration for the metrics collector.
        """
        self.config = config
    
    
    def forward(self, *args: List[Dict[str, Any]]) -> ImageMetricsOutput:
        metrics_enabled = self.config["train"] if self.training else self.config["eval"]
            
        collected_dict = args[0]
        # print(collected_dict.keys())
        for arg in args[1:]:
            if isinstance(arg, dict):
                collected_dict.update(arg)
            
        output_metrics = dict()
        
        assert "x" in collected_dict, "Input tensor 'x' is required."
        assert "x_hat" in collected_dict, "Reconstructed tensor 'x_hat' is required."
        num_pixels = collected_dict["x"].shape[0] * collected_dict["x"].shape[2] * collected_dict["x"].shape[3]

        if "likelihoods" in collected_dict:
            total_bits = sum(
                torch.log(likelihood).sum() / (-math.log(2))
                for likelihood in collected_dict["likelihoods"].values()
            )
        elif "strings" in collected_dict:
            total_bits = sum(len(s) * 8 if isinstance(s, bytes) else len(s[0]) * 8 for s in collected_dict["strings"])
        else:
            raise ValueError("Either likelihoods or strings must be provided.")
        
        output_metrics["bpp"] = total_bits / num_pixels
        
        if "MSE" in metrics_enabled:
            mse_loss = F.mse_loss(collected_dict['x_hat'], collected_dict['x'])
            psnr = 10 * torch.log10(1 / mse_loss)
            output_metrics.update({
                "mse_loss": mse_loss,
                "psnr": psnr
            })
            

        if "MS_SSIM" in metrics_enabled:
            ms_ssim = ms_ssim_func(
                collected_dict["x_hat"], 
                collected_dict["x"], 
                data_range=1.0, 
                size_average=True
            )
            ms_ssim_loss = 1 - ms_ssim
            output_metrics.update({
                "ms_ssim_loss": ms_ssim_loss,
                "ms_ssim": ms_ssim
            })
        
        # TODO: under test
        # if "KID" in metrics_enabled and not self.training:
        #     self.kid_metric.forward()
        
        # if "FID" in metrics_enabled and not self.training:
        #     self.fid_metric.forward()
        
        if "LPIPS" in metrics_enabled:
            output_metrics["lpips"] = lpips_func(
                collected_dict["x_hat"].clamp(0, 1), 
                collected_dict["x"], 
                net_type='alex', 
                normalize=True
            )
        
        if "DISTS" in metrics_enabled:
            output_metrics["dists"] = dists_func(
                collected_dict["x_hat"], 
                collected_dict["x"], 
                reduction='mean',
            )
            
        return ImageMetricsOutput(**output_metrics)
    
    # def state_