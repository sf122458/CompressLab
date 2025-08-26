import torch
import torch.nn.functional as F
import math
from torch import Tensor
from typing import List, Dict, Any, Union
from dataclasses import dataclass, field
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from lightning import LightningModule


@dataclass
class ImageMetricsOutput:
    bpp: Union[float, torch.Tensor]
    mse_loss: torch.Tensor = field(default=None)
    psnr: float = field(default=None)
    ms_ssim_loss: torch.Tensor = field(default=None)
    ms_ssim: float = field(default=None)
    lpips: torch.Tensor = field(default=None)
    dists: torch.Tensor = field(default=None)
    kid: float = field(default=None)
    fid: float = field(default=None)
        

class MetricsCollector(LightningModule):
    def __init__(
        self, 
    ):
        super().__init__()
        
        self.initialized = False
        
    def setup(self):
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(self.device)
        self.dists = DeepImageStructureAndTextureSimilarity(reduction='mean').to(self.device)
        self.fid = FrechetInceptionDistance(normalize=True).to(self.device)
        self.kid = KernelInceptionDistance(normalize=True).to(self.device)
    
    def forward(self, imgs: Tensor, recons: Tensor, 
                likelihoods: Dict[str, Tensor] = None, 
                strings: List[Any] = None,
                mse: bool = False,
                psnr: bool = False,
                ms_ssim: bool = False,
                lpips: bool = False,
                dists: bool = False,
                kid: bool = False,
                fid: bool = False,
                ) -> ImageMetricsOutput:
        if self.initialized is False:   # This avoid saving the pretrained model weights in checkpoints
            self.setup()
            self.initialized = True
        
        recons = recons.clamp(0, 1)
        
        output_metrics = dict()
        
        num_pixels = imgs.shape[0] * imgs.shape[2] * imgs.shape[3]
        
        if likelihoods is not None:
            total_bits = sum(
                torch.log(likelihood).sum() / (-math.log(2))
                for likelihood in likelihoods.values()
            )
        elif strings is not None:
            total_bits = 0
            for s in strings:
                while isinstance(s, list):
                    s = s[0]
                total_bits += len(s) * 8
        else:
            raise ValueError("Either likelihoods or strings must be provided.")
        
        output_metrics["bpp"] = total_bits / num_pixels
        
        if mse or psnr:
            mse_loss = F.mse_loss(recons, imgs)
            psnr = 10 * torch.log10(1 / mse_loss)
            output_metrics.update({
                "mse_loss": mse_loss,
                "psnr": psnr
            })
            

        if ms_ssim:
            ms_ssim = self.ms_ssim(
                recons,
                imgs, 
            )
            ms_ssim_loss = 1 - ms_ssim
            output_metrics.update({
                "ms_ssim_loss": ms_ssim_loss,
                "ms_ssim": ms_ssim
            })
        
        if lpips:
            output_metrics["lpips"] = self.lpips(recons, imgs)
        
        if dists:
            output_metrics["dists"] = self.dists(recons, imgs)
            
        if kid:
            self._update_patch(imgs, recons, self.kid)
            try:
                output_metrics["kid"] = self.kid.compute()[0]
            except Exception as e:
                output_metrics["kid"] = torch.tensor(float('nan'))
        
        if fid:
            self._update_patch(imgs, recons, self.fid)
            try:
                output_metrics["fid"] = self.fid.compute()
            except Exception as e:
                output_metrics["fid"] = torch.tensor(float('nan'))
            
        return ImageMetricsOutput(**output_metrics)
    
    def reset(self):
        self.ms_ssim.reset()
        self.lpips.reset()
        self.dists.reset()
        self.fid.reset()
        self.kid.reset()

    def _update_patch(self, input_images: Tensor, pred: Tensor, 
                     metrics_fn: Union[KernelInceptionDistance, FrechetInceptionDistance], 
                     patch_size=256):
        real = F.unfold(input_images, kernel_size=patch_size, stride=patch_size)\
            .permute(0, 2, 1)\
            .reshape(-1, 3, patch_size, patch_size)
        fake = F.unfold(pred, kernel_size=patch_size, stride=patch_size)\
            .permute(0, 2, 1)\
            .reshape(-1, 3, patch_size, patch_size)
        
        metrics_fn.update(real, real=True)
        metrics_fn.update(fake, real=False)
        
        patch_count = real.shape[0]

        H, W = input_images.shape[2], input_images.shape[3]
        if H >= 1.5 * patch_size and W >= 1.5 * patch_size:
            real = F.unfold(
                    input_images[:, :, patch_size // 2 :, patch_size // 2 :],
                    kernel_size=patch_size,
                    stride=patch_size,
                )\
                .permute(0, 2, 1)\
                .reshape(-1, 3, patch_size, patch_size)
            fake = F.unfold(
                    pred[:, :, patch_size // 2 :, patch_size // 2 :],
                    kernel_size=patch_size,
                    stride=patch_size,
                )\
                .permute(0, 2, 1)\
                .reshape(-1, 3, patch_size, patch_size)
                
            patch_count += real.shape[0]
            
            metrics_fn.update(real, real=True)
            metrics_fn.update(fake, real=False)
            
        return patch_count