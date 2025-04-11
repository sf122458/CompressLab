import lightning as L
from compressai.models import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
import torch
import math
import torch.nn as nn
from copy import deepcopy

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

        self.model_wrapper: nn.ModuleDict[str, CompressionModel] = nn.ModuleDict([])

        if isinstance(self.lmbda, list):
            for idx in range(len(self.lmbda)):
                self.model_wrapper[f"codec_{idx}"] = deepcopy(model)
            del model
        else:
            self.lmbda = [lmbda]
            self.model_wrapper["codec"] = model
    
    def compress(self, x):
        raise NotImplementedError("Compress function is not implemented.")
    
    def decompress(self, x):
        raise NotImplementedError("Decompress function is not implemented.")

    def training_step(self, batch, batch_idx):
        optimizer, aux_optimizer = self.optimizers()
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        x = batch
        N, _, H, W = x.shape
        for lmbda, (model_name, model_instance) in zip(self.lmbda, self.model_wrapper.items()):
            out = model_instance.forward(x)
            distortion_loss = torch.nn.functional.mse_loss(out["x_hat"], x)
            bpp_loss = \
                sum(
                    torch.log(likelihoods).sum() / (-math.log(2) * N * H * W)
                    for likelihoods in out["likelihoods"].values()
                )
            psnr = 10 * torch.log10(1 / distortion_loss).item()
            loss = lmbda * distortion_loss + bpp_loss
            aux_loss = model_instance.aux_loss()
            self.manual_backward(loss)
            self.manual_backward(aux_loss)

            torch.nn.utils.clip_grad_norm_(model_instance.parameters(), 1.0)

            if model_name == "codec_0" or model_name == "codec":
                self.log_dict({"loss": loss, 
                        "bpp": bpp_loss,
                        "PSNR": psnr}, prog_bar=True, on_step=True, on_epoch=False, logger=False)
        
            self.log_dict({
                f"train/{model_name}.loss": loss,
                f"train/{model_name}.bpp": bpp_loss,
                f"train/{model_name}.psnr": psnr,
            }, on_epoch=True, logger=True, sync_dist=True, on_step=False)

        optimizer.step()
        aux_optimizer.step()
        

    def validation_step(self, batch, batch_idx):
        x = batch
        N, _, H, W = x.shape
        for model_name, model_instance in self.model_wrapper.items():
            out = model_instance.forward(x)
            distortion_loss = torch.nn.functional.mse_loss(out["x_hat"], x)
            psnr = 10 * torch.log10(1 / distortion_loss)
            bpp_loss = \
                sum(
                    torch.log(likelihoods).sum() / (-math.log(2) * N * H * W)
                    for likelihoods in out["likelihoods"].values()
                )
            
            self.log_dict({
                f"val/{model_name}.bpp": bpp_loss,
                f"val/{model_name}.psnr": psnr
            }, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
    # def on_validation_epoch_end(self):
    #     return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        self.compress(batch)


    def configure_optimizers(self):
        parameters = []
        aux_parameters = []
        for model_name, model_instance in self.model_wrapper.items():
            parameters += [p for n, p in model_instance.named_parameters() if p.requires_grad and not n.endswith(".quantiles")]
            aux_parameters += [p for n, p in model_instance.named_parameters() if p.requires_grad and n.endswith(".quantiles")]
        # TODO
        optimizer = torch.optim.Adam(parameters, lr=self.lr)
        aux_optimizer = torch.optim.Adam(aux_parameters, lr=1e-3)
        return optimizer, aux_optimizer
        

