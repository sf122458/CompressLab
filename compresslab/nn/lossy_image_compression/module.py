import lightning as L
from compressai.models import CompressionModel
from compresslab.utils.logger import MetricLogger
import torch
import math
import torch.nn as nn
import numpy as np
from copy import deepcopy
from pytorch_msssim import ms_ssim
from typing import Dict, List, Any

class CompressAILightningModule(L.LightningModule):
    def __init__(self, 
                 model: CompressionModel,
                 ext_params: Dict[str, Any] = None
                 ):
        super().__init__()

        # As there are two optimizers, we need to set the automatic optimization to False
        self.automatic_optimization = False

        if "lmbda" not in ext_params.keys():
            raise ValueError("lmbda is required in ext_params")

        self.lmbda = ext_params.get("lmbda")
        self.lr = ext_params.get("lr", 1e-4)
        self.distortion = ext_params.get("distortion", "mse")
        assert self.distortion in ["mse", "ms-ssim"], f"invalid distortion: {self.distortion}, only mse and ms-ssim are supported"


        self.model_wrapper: nn.ModuleDict[str, CompressionModel] = nn.ModuleDict({})
        
        if isinstance(self.lmbda, list):
            for idx in range(len(self.lmbda)):
                self.model_wrapper[f"codec_{idx}"] = deepcopy(model)
            del model
        else:
            self.lmbda = [self.lmbda]
            self.model_wrapper["codec"] = model
    

    def training_step(self, batch, batch_idx):
        optimizer, aux_optimizer = self.optimizers()
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        x = batch
        N, _, H, W = x.shape
        for lmbda, (model_name, model_instance) in zip(self.lmbda, self.model_wrapper.items()):
            out = model_instance.forward(x)
            
            bpp_loss = \
                sum(
                    torch.log(likelihoods).sum() / (-math.log(2) * N * H * W)
                    for likelihoods in out["likelihoods"].values()
                )
            
            mse_loss = torch.nn.functional.mse_loss(out["x_hat"], x)
            psnr = 10 * torch.log10(1. / mse_loss)
            ms_ssim_loss = ms_ssim(out["x_hat"], x, data_range=1)
            
            if self.distortion == "mse":
                distortion_loss = mse_loss * 255 ** 2
            else:
                distortion_loss = 1 - ms_ssim_loss

            loss = lmbda * distortion_loss + bpp_loss
            aux_loss = model_instance.aux_loss()
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(model_instance.parameters(), 1.0)
            self.manual_backward(aux_loss)

            # show metrics of the first model on the progress bar
            if model_name == "codec_0" or model_name == "codec":
                self.log_dict({"loss": loss, 
                        "bpp": bpp_loss,
                        "PSNR": psnr,
                        "MS-SSIM": ms_ssim_loss}, prog_bar=True, on_step=True, on_epoch=False, logger=False)
        
            self.log_dict({
                f"train/{model_name}.loss": loss,
                f"train/{model_name}.bpp": bpp_loss,
                f"train/{model_name}.psnr": psnr,
                f"train/{model_name}.ms-ssim": ms_ssim_loss,
            }, on_epoch=False, logger=True, sync_dist=True, on_step=True)

        optimizer.step()
        aux_optimizer.step()
        

    def validation_step(self, batch, batch_idx):
        x = batch
        N, _, H, W = x.shape
        for model_name, model_instance in self.model_wrapper.items():
            out = model_instance.forward(x)
            mse_loss = torch.nn.functional.mse_loss(out["x_hat"], x)
            psnr = 10 * torch.log10(1 / mse_loss)
            bpp_loss = \
                sum(
                    torch.log(likelihoods).sum() / (-math.log(2) * N * H * W)
                    for likelihoods in out["likelihoods"].values()
                )
            
            ms_ssim_loss = ms_ssim(out["x_hat"], x, data_range=1)
            
            self.log_dict({
                f"val/{model_name}.bpp": bpp_loss,
                f"val/{model_name}.psnr": psnr,
                f"val/{model_name}.ms-ssim": ms_ssim_loss,
            }, on_step=False, on_epoch=True, logger=True, sync_dist=True)

    def on_test_start(self):
        self.metric = MetricLogger(save_dir=self.trainer.default_root_dir)
        for model_name, model_instance in self.model_wrapper.items():
            model_instance.update()

    def test_step(self, batch, batch_idx):
        for model_name, model_instance in self.model_wrapper.items():
            with self.metric.timer(model_name, "time_compress") as timer:
                out_compress = model_instance.compress(batch)
            with self.metric.timer(model_name, "time_decompress") as timer:
                out_decompress = model_instance.decompress(out_compress["strings"], out_compress["shape"])
            mse_loss = torch.nn.functional.mse_loss(out_decompress["x_hat"], batch)
            psnr = 10 * torch.log10(1 / mse_loss).item()
            ms_ssim_loss = ms_ssim(out_decompress["x_hat"], batch, data_range=1).item()
            bpp = sum(len(strings[0]) for strings in out_compress["strings"]) / np.prod(batch.shape[-2:]) * 8
            self.metric.log(model_name, 
                            {
                                "bpp":bpp, 
                                "psnr":psnr,
                                "ms-ssim":ms_ssim_loss,
                             })

    def on_test_end(self):
        self.metric.save()

    def configure_optimizers(self):
        parameters = []
        aux_parameters = []
        for model_name, model_instance in self.model_wrapper.items():
            parameters += [p for n, p in model_instance.named_parameters() if p.requires_grad and not n.endswith(".quantiles")]
            aux_parameters += [p for n, p in model_instance.named_parameters() if p.requires_grad and n.endswith(".quantiles")]
        optimizer = torch.optim.Adam(parameters, lr=self.lr)
        aux_optimizer = torch.optim.Adam(aux_parameters, lr=1e-3)
        return optimizer, aux_optimizer
        

