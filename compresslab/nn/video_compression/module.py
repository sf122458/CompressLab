import lightning as L
from compressai.models import CompressionModel
from compresslab.utils.logger import MetricLogger
import torch
import math
import torch.nn as nn
import numpy as np
from copy import deepcopy
from compresslab.nn.video_compression.compressai_impl.utils import RateDistortionLoss
from typing import Dict, Any
from pytorch_msssim import ms_ssim

class DVCLightingModule(L.LightningModule):
    """
    Support models similar to DVC, which contain motion codecs and residual/contextual codecs.
    """
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
    
    def on_train_batch_start(self, batch, batch_idx):
        if self.global_step == self.lr_decay_interval:
            for optimizer in self.optimizers():
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= self.lr_decay

    def training_step(self, batch, batch_idx):
        optimizer, aux_optimizer = self.optimizers()
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        input_frame, ref_frame = batch
        N, _, H, W = input_frame.shape

        for lmbda, (model_name, model_instance) in zip(self.lmbda, self.model_wrapper.items()):
            out = model_instance.forward(input_frame, ref_frame)

            bpp_mv_y = torch.log(out["likelihoods"]["y_mv"]).sum() / (-math.log(2) * N * H * W)
            bpp_y = torch.log(out["likelihoods"]["y"]).sum() / (-math.log(2) * N * H * W)
            bpp_mv_z = torch.log(out["likelihoods"]["z_mv"]).sum() / (-math.log(2) * N * H * W)
            bpp_z = torch.log(out["likelihoods"]["z"]).sum() / (-math.log(2) * N * H * W)

            bpp_loss = bpp_mv_z + bpp_mv_y + bpp_z + bpp_y
            
            mse_loss = torch.nn.functional.mse_loss(out["recon_frame"], input_frame)

            psnr = 10 * torch.log10(1. / mse_loss)

            #TODO
            distortion_loss = mse_loss

            loss = lmbda * distortion_loss + bpp_loss
            aux_loss = model_instance.aux_loss()
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(model_instance.parameters(), 0.5)
            self.manual_backward(aux_loss)

            if model_name == "codec_0" or model_name == "codec":
                self.log_dict(dict(
                    loss=loss,
                    bpp=bpp_loss,
                    bpp_mv_y=bpp_mv_y,
                    bpp_y=bpp_y,
                    psnr=psnr,
                ), prog_bar=True, on_step=True, on_epoch=False, logger=False)

            self.log_dict({
                f"train/{model_name}.loss": loss,
                f"train/{model_name}.bpp": bpp_loss,
                f"train/{model_name}.bpp_mv_y": bpp_mv_y,
                f"train/{model_name}.bpp_mv_z": bpp_mv_z,
                f"train/{model_name}.bpp_y": bpp_y,
                f"train/{model_name}.bpp_z": bpp_z,
                f"train/{model_name}.psnr": psnr,
            }, on_epoch=False, logger=True, sync_dist=True, on_step=True)
        
        optimizer.step()
        aux_optimizer.step()

    def on_validation_start(self):
        for model_name, model_instance in self.model_wrapper.items():
            model_instance.update()

    def validation_step(self, batch, batch_idx):
        for lmbda, (model_name, model_instance) in zip(self.lmbda, self.model_wrapper.items()):

            input_frames_dict, ref_frame_dict, ref_bpp_dict, ref_psnr_dict, ref_msssim_dict = batch

            input_frames = input_frames_dict[lmbda]
            ref_frame = ref_frame_dict[lmbda].squeeze(0)
            seqlen = input_frames.shape[1]

            avg_bpp = ref_bpp_dict[lmbda]
            avg_psnr = ref_psnr_dict[lmbda]
            avg_msssim = ref_msssim_dict[lmbda]

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
            input_frames_dict, ref_frame_dict, ref_bpp_dict, ref_psnr_dict, ref_msssim_dict = batch
            input_frames = input_frames_dict[lmbda].squeeze(0)
            ref_bpp = ref_bpp_dict[lmbda]
            ref_psnr = ref_psnr_dict[lmbda]
            ref_msssim = ref_msssim_dict[lmbda]
            ref_frame = ref_frame_dict[lmbda]
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
        optimizer = torch.optim.Adam(parameters, lr=self.lr)
        aux_optimizer = torch.optim.Adam(aux_parameters, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_decay_interval, gamma=self.lr_decay)
        return optimizer, aux_optimizer


class CompressAILightningModule(L.LightningModule):
    """
    In CompressAI implementation, the models contain a image codec to compress the I frame, 
    and a video frame codec to compress the P frame, which is different from the training stategy of DVC.
    """
    def __init__(self, 
                 model: CompressionModel,
                 ext_params: Dict[str, Any] = {}):
        super().__init__()

        # As there are two optimizers, we need to set the automatic optimization to False
        self.automatic_optimization = False

        self.lmbda = ext_params.get("lmbda", 1)
        self.lr = ext_params.get("lr", 1e-4)

        self.model_wrapper: nn.ModuleDict[str, CompressionModel] = nn.ModuleDict({})
        
        self.loss_func = RateDistortionLoss()

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
        