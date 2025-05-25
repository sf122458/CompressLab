import lightning as L
from typing import Union, List
from compressai.models import CompressionModel
import torch, math
import torch.nn as nn
from copy import deepcopy
from typing import Dict, Any

class VideoLightingModule(L.LightningModule):
    def __init__(self,
                 model: CompressionModel,
                 ext_params: Dict[str, Any] = {}
                 ):
        super().__init__()

        self.automatic_optimization = False

        if "lmbda" not in ext_params.keys():
            raise ValueError("lmbda is required in ext_params")

        self.lmbda = ext_params.get("lmbda")
        self.lr = ext_params.get("lr", 1e-4)
        self.lr_decay = ext_params.get("lr_decay", 0.1)
        self.lr_decay_interval = ext_params.get("lr_decay_interval", 1_800_000)


        self.model_wrapper: nn.ModuleDict[str, CompressionModel] = nn.ModuleDict({})

        if isinstance(self.lmbda, list):
            for idx in range(len(self.lmbda)):
                self.model_wrapper[f"codec_{idx}"] = deepcopy(model)
            del model
        else:
            self.lmbda = [self.lmbda]
            self.model_wrapper["codec"] = model

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        input_frame, ref_frame = batch
        N, _, H, W = input_frame.shape

        for lmbda, (model_name, model_instance) in zip(self.lmbda, self.model_wrapper.items()):
            out = model_instance.forward(input_frame, ref_frame)

            bpp_loss = \
                sum(
                    torch.log(likelihoods).sum() / (-math.log(2) * N * H * W)
                    for likelihoods in out["likelihoods"].values()
                )

            bpp_mv = torch.log(out["likelihoods"]["mv"]).sum() / (-math.log(2) * N * H * W)
            bpp_res = torch.log(out["likelihoods"]["res"]).sum() / (-math.log(2) * N * H * W)
            
            mse_loss = torch.nn.functional.mse_loss(out["recon_frame"], input_frame)
            warp_loss = torch.nn.functional.mse_loss(out["warp_frame"], input_frame)
            inter_loss = torch.nn.functional.mse_loss(out["prediction"], input_frame)

            psnr=10 * torch.log10(1. / mse_loss)
            warp_psnr=10 * torch.log10(1. / warp_loss)
            inter_psnr=10 * torch.log10(1. / inter_loss)

            #TODO
            # if self.global_step < 500_000:
            #     distortion_loss = mse_loss + warp_loss + inter_loss
            # else:
            #     distortion_loss = mse_loss

            distortion_loss = mse_loss + warp_loss + inter_loss

            loss = lmbda * distortion_loss + bpp_loss
            aux_loss = model_instance.aux_loss()
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(model_instance.parameters(), 0.5)
            self.manual_backward(aux_loss)

            if model_name == "codec_0" or model_name == "codec":
                self.log_dict(dict(
                    loss=loss,
                    bpp=bpp_loss,
                    bpp_mv=bpp_mv,
                    bpp_res=bpp_res,
                    psnr=psnr,
                    warp_psnr=warp_psnr,
                    inter_psnr=inter_psnr,
                ), prog_bar=True, on_step=True, on_epoch=False, logger=False)

            self.log_dict({
                f"train/{model_name}.loss": loss,
                f"train/{model_name}.bpp": bpp_loss,
                f"train/{model_name}.bpp_mv": bpp_mv,
                f"train/{model_name}.bpp_res": bpp_res,
                f"train/{model_name}.psnr": psnr,
                f"train/{model_name}.warp_psnr": warp_psnr,
                f"train/{model_name}.inter_psnr": inter_psnr,
            }, on_epoch=False, logger=True, sync_dist=True, on_step=True)
        
        optimizer.step()


    # def validation_step(self, batch, batch_idx):
    #     input_frames, ref_frame, ref_bpp, ref_psnr, ref_msssim = batch
    #     seqlen = input_frames.shape[1]

    #     #TODO: logging

    #     for lmbda, (model_name, model_instance) in zip(self.lmbda, self.model_wrapper.items()):
    #         for i in range(seqlen):
    #             input_frame = input_frames[:, i, :, :, :]
    #             out = model_instance.forward(input_frame, ref_frame)

    #             bpp_loss = \
    #             sum(
    #                 torch.log(likelihoods).sum() / (-math.log(2) * N * H * W)
    #                 for likelihoods in out["likelihoods"].values()
    #             )

    #             bpp_mv = torch.log(out["likelihoods"]["mv"]).sum() / (-math.log(2) * N * H * W)
    #             bpp_res = torch.log(out["likelihoods"]["res"]).sum() / (-math.log(2) * N * H * W)
                
    #             mse_loss = torch.nn.functional.mse_loss(out["recon_frame"], input_frame)
    #             warp_loss = torch.nn.functional.mse_loss(out["warp_frame"], input_frame)
    #             inter_loss = torch.nn.functional.mse_loss(out["prediction"], input_frame)

    #             psnr=10 * torch.log10(1. / mse_loss)
    #             warp_psnr=10 * torch.log10(1. / warp_loss)
    #             inter_psnr=10 * torch.log10(1. / inter_loss)


    # def test_step(self, batch, batch_idx):
    #     pass
    

    def configure_optimizers(self):
        parameters = []
        aux_parameters = []
        for model_name, model_instance in self.model_wrapper.items():
            parameters += [p for n, p in model_instance.named_parameters() if p.requires_grad and not n.endswith(".quantiles")]
            aux_parameters += [p for n, p in model_instance.named_parameters() if p.requires_grad and n.endswith(".quantiles")]
        optimizer = torch.optim.Adam(
            [{
                "params": parameters,
                "lr": self.lr
            },
            {
                "params": aux_parameters,
                "lr": 1e-3
            }])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_decay_interval, gamma=self.lr_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }