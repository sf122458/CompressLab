import lightning as L
from compressai.models import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
import torch
import math
import torch.nn as nn

class CompressAILightningModule(L.LightningModule, CompressionModel):
    def __init__(self, 
                 lr: float = 1e-4,
                 lmbda: float = 0.1):
        super().__init__()

        # As there are two optimizers, we need to set the automatic optimization to False
        self.automatic_optimization = False

        self.lmbda = lmbda
        self.lr = lr

    def __init_subclass__(cls):
        return super().__init_subclass__()

    def forward(self, x):
        raise NotImplementedError("Forward function is not implemented.")
    
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
        out = self.forward(x)
        distortion_loss = torch.nn.functional.mse_loss(out["x_hat"], x)
        bpp_loss = \
            sum(
                torch.log(likelihoods).sum() / (-math.log(2) * N * H * W)
                for likelihoods in out["likelihoods"].values()
            )
        psnr = 10 * torch.log10(1 / distortion_loss).item()
        loss = self.lmbda * distortion_loss + bpp_loss
        aux_loss = self.aux_loss()
        self.manual_backward(loss)
        self.manual_backward(aux_loss)

        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        optimizer.step()
        aux_optimizer.step()
        self.log_dict({"loss": loss, 
                       "bpp": bpp_loss,
                       "PSNR": psnr}, prog_bar=True, on_step=True, on_epoch=False, logger=False)
        
        self.log_dict({
            "train/loss": loss,
            "train/bpp": bpp_loss,
            "train/psnr": psnr,
        }, on_epoch=True, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x = batch
        N, _, H, W = x.shape
        out = self.forward(x)
        distortion_loss = torch.nn.functional.mse_loss(out["x_hat"], x)
        psnr = 10 * torch.log10(1 / distortion_loss)
        bpp_loss = \
            sum(
                torch.log(likelihoods).sum() / (-math.log(2) * N * H * W)
                for likelihoods in out["likelihoods"].values()
            )
        
    # def on_validation_epoch_end(self):
    #     return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        self.compress(batch)


    def configure_optimizers(self):
        parameters = [p for n, p in self.named_parameters() if p.requires_grad and not n.endswith(".quantiles")]
        aux_parameters = [p for n, p in self.named_parameters() if p.requires_grad and n.endswith(".quantiles")]
        # TODO
        optimizer = torch.optim.Adam(parameters, lr=self.lr)
        aux_optimizer = torch.optim.Adam(aux_parameters, lr=1e-3)
        return optimizer, aux_optimizer
        

