import lightning as L
from compressai.models import CompressionModel
from compresslab.utils.logger import MetricLogger
import torch
import math
import torch.nn as nn
import numpy as np
from copy import deepcopy
from compresslab.nn.video_compression.utils import RateDistortionLoss

class CompressAILightningModule(L.LightningModule):
    def __init__(self, 
                 model: CompressionModel,
                 lr: float = 1e-4,
                 lmbda: float = 0.1):
        super().__init__()

        # As there are two optimizers, we need to set the automatic optimization to False
        self.automatic_optimization = False

        self.lmbda = lmbda
        self.lr = lr

        self.model_wrapper: nn.ModuleDict[str, CompressionModel] = nn.ModuleDict({})
        
        self.loss_func = RateDistortionLoss()

        if isinstance(self.lmbda, list):
            for idx in range(len(self.lmbda)):
                self.model_wrapper[f"codec_{idx}"] = deepcopy(model)
            del model
        else:
            self.lmbda = [lmbda]
            self.model_wrapper["codec"] = model
    

    def training_step(self, batch, batch_idx):
        optimizer, aux_optimizer = self.optimizers()
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        d = batch

        for lmbda, (model_name, model_instance) in zip(self.lmbda, self.model_wrapper.items()):
            out = model_instance.forward(d)

            out_criterion = self.loss_func.forward(out, d, lmbda)
            aux_loss_sum = sum(aux_loss for aux_loss in model_instance.aux_loss())

            self.manual_backward(out_criterion["loss"])
            torch.nn.utils.clip_grad_norm_(model_instance.parameters(), 1.0)
            self.manual_backward(aux_loss_sum)

            # show metrics of the first model on the progress bar
            if model_name == "codec_0" or model_name == "codec":
                self.log_dict({
                    "loss": out_criterion["loss"], 
                    "bpp": out_criterion["bpp_loss"],
                    "PSNR": out_criterion["psnr"]
                }, prog_bar=True, on_step=True, on_epoch=False, logger=False)
        
            self.log_dict({
                f"train/{model_name}.loss": out_criterion["loss"],
                f"train/{model_name}.bpp": out_criterion["bpp_loss"],
                f"train/{model_name}.psnr": out_criterion["psnr"]
            }, on_epoch=False, logger=True, sync_dist=True, on_step=True)

        optimizer.step()
        aux_optimizer.step()
        

    def validation_step(self, batch, batch_idx):
        x = batch
        for model_name, model_instance in self.model_wrapper.items():
            out = model_instance.forward(x)

            out_criterion = self.loss_func.forward(out, x)

            self.log_dict({
                f"val/{model_name}.bpp": out_criterion["bpp_loss"],
                f"val/{model_name}.psnr": out_criterion["psnr"],
            }, on_step=False, on_epoch=True, logger=True, sync_dist=True)

    def on_test_start(self):
        return
        self.metric = MetricLogger(save_dir=self.trainer.default_root_dir)
        for model_name, model_instance in self.model_wrapper.items():
            model_instance.update()

    def test_step(self, batch, batch_idx):
        return
        for model_name, model_instance in self.model_wrapper.items():
            with self.metric.timer(model_name, "time_compress") as timer:
                out_compress = model_instance.compress(batch)
            with self.metric.timer(model_name, "time_decompress") as timer:
                out_decompress = model_instance.decompress(out_compress["strings"], out_compress["shape"])
            distortion_loss = torch.nn.functional.mse_loss(out_decompress["x_hat"], batch)
            psnr = 10 * torch.log10(1 / distortion_loss).item()
            bpp = sum(len(strings[0]) for strings in out_compress["strings"]) / np.prod(batch.shape[-2:]) * 8
            self.metric.log(model_name, 
                            {
                                "bpp":bpp, 
                                "psnr":psnr
                             })

    def on_test_end(self):
        return
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
        