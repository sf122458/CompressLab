import lightning as L
from typing import Union, List
from compressai.models import CompressionModel
from compresslab.utils.logger import MetricLogger
import torch, math
import torch.nn as nn
from copy import deepcopy
from typing import Dict, Any
import numpy as np
from pytorch_msssim import ms_ssim

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

    def on_validation_start(self):
        for model_name, model_instance in self.model_wrapper.items():
            model_instance.update()

    def validation_step(self, batch, batch_idx):
        for lmbda, (model_name, model_instance) in zip(self.lmbda, self.model_wrapper.items()):

            input_frames, ref_frame, ref_bpp, ref_psnr, ref_msssim = batch
            seqlen = input_frames.shape[1]

            avg_bpp = ref_bpp
            avg_psnr = ref_psnr
            avg_msssim = ref_msssim

            for i in range(seqlen):
                input_frame = input_frames[:, i, :, :, :]
                out_compress = model_instance.compress(input_frame, ref_frame)

                out_decompress = model_instance.decompress(ref_frame, out_compress["strings"], out_compress["shape"])

                recon_frame = out_decompress["recon_frame"]

                bpp = sum(len(strings[0]) for strings in out_compress["strings"]) / np.prod(input_frame.shape[-2:]) * 8
                mse_loss = torch.nn.functional.mse_loss(recon_frame, input_frame)
                psnr = 10 * torch.log10(1. / mse_loss).item()
                ms_ssim_loss = ms_ssim(recon_frame, input_frame, data_range=1).item()

                avg_bpp += bpp
                avg_psnr += psnr
                avg_msssim += ms_ssim_loss

                ref_frame = recon_frame

            self.log_dict({
                f"val/{model_name}.bpp": avg_bpp / (seqlen + 1),
                f"val/{model_name}.psnr": avg_psnr / (seqlen + 1),
                f"val/{model_name}.ms-ssim": avg_msssim / (seqlen + 1),
            }, on_step=False, on_epoch=True, logger=True, sync_dist=True)

    def on_test_start(self):
        self.metric = MetricLogger(save_dir=self.trainer.default_root_dir)
        for model_name, model_instance in self.model_wrapper.items():
            model_instance.update()


    def test_step(self, batch, batch_idx):
        for lmbda, (model_name, model_instance) in zip(self.lmbda, self.model_wrapper.items()):
            input_frames, ref_frame, ref_bpp, ref_psnr, ref_msssim = batch
            seqlen = input_frames.shape[1]

            self.metric.log(model_name, {
                "bpp": ref_bpp,
                "psnr": ref_psnr,
                "ms-ssim": ref_msssim,
            })

            for i in range(seqlen):
                input_frame = input_frames[:, i, :, :, :]
                with self.metric.timer(model_name, "time_compress") as timer:
                    out_compress = model_instance.compress(input_frame, ref_frame)

                with self.metric.timer(model_name, "time_decompress") as timer:
                    out_decompress = model_instance.decompress(ref_frame, out_compress["strings"], out_compress["shape"])

                recon_frame = out_decompress["recon_frame"]

                bpp = sum(len(strings[0]) for strings in out_compress["strings"]) / np.prod(input_frame.shape[-2:]) * 8
                mse_loss = torch.nn.functional.mse_loss(recon_frame, input_frame)
                psnr = 10 * torch.log10(1. / mse_loss).item()
                ms_ssim_loss = ms_ssim(recon_frame, input_frame, data_range=1).item()

                ref_frame = recon_frame

                self.metric.log(model_name,{
                    "bpp": bpp,
                    "psnr": psnr,
                    "ms-ssim": ms_ssim_loss,
                })

    def on_test_end(self):
        self.metric.save()

    

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