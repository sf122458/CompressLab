import lightning as L
from compresslab.core.models import CompressionModel
from compresslab.utils.logger import MetricLogger
from compresslab.nn.video_compression.abc import *
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from compresslab.nn.video_compression.compressai_impl.utils import RateDistortionLoss
from typing import Dict, Any, Union
from pytorch_msssim import ms_ssim

class PFrameCodecLightingModule(L.LightningModule):
    """
    A LightningModule for training P-frame codecs.
    """
    def __init__(self,
                 model: PFrameCodec,
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
        self.progressive_training = ext_params.get("progressive_training", None)

        if self.progressive_training is not None:
            assert isinstance(self.progressive_training, list) and len(self.progressive_training) == 3

        self.model_wrapper: nn.ModuleDict[str, PFrameCodec] = nn.ModuleDict({})

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

        for lmbda, (model_name, model_instance) in zip(self.lmbda, self.model_wrapper.items()):
            model_instance: Union[PFrameCodec, CompressionModel]
            out = model_instance.forward(PFrameForwardInput(
                input_frame=input_frame,
                refer_frame=ref_frame
            ))

            #TODO
            distortion_loss = out.mse_loss
            if self.progressive_training is None:
                bpp_loss = out.likelihoods.bpp_z_mv \
                    + out.likelihoods.bpp_y_mv \
                    + out.likelihoods.bpp_z \
                    + out.likelihoods.bpp_y
                
            else:
                if self.global_step < self.progressive_training[0]:
                    bpp_loss = out.likelihoods.bpp_y_mv + out.likelihoods.bpp_z_mv
                elif self.global_step < self.progressive_training[1]:
                    bpp_loss = 0
                elif self.global_step < self.progressive_training[2]:
                    bpp_loss = out.likelihoods.bpp_y + out.likelihoods.bpp_z
                else:
                    bpp_loss = out.likelihoods.bpp_y_mv \
                        + out.likelihoods.bpp_z_mv \
                        + out.likelihoods.bpp_y \
                        + out.likelihoods.bpp_z

            
            loss = lmbda * distortion_loss + bpp_loss

            aux_loss = model_instance.aux_loss()
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(model_instance.parameters(), 0.5)
            self.manual_backward(aux_loss)

            if model_name == "codec_0" or model_name == "codec":
                self.log_dict(dict(
                    loss=loss,
                    bpp=out.bpp,
                    bpp_y_mv=out.likelihoods.bpp_y_mv,
                    bpp_y=out.likelihoods.bpp_y,
                    psnr=out.psnr,
                ), prog_bar=True, on_step=True, on_epoch=False, logger=False)

            self.log_dict({
                f"train/{model_name}.loss": loss,
                f"train/{model_name}.bpp": out.bpp,
                f"train/{model_name}.bpp_y_mv": out.likelihoods.bpp_y_mv,
                f"train/{model_name}.bpp_z_mv": out.likelihoods.bpp_z_mv,
                f"train/{model_name}.bpp_y": out.likelihoods.bpp_y,
                f"train/{model_name}.bpp_z": out.likelihoods.bpp_z,
                f"train/{model_name}.psnr": out.psnr,
            }, on_epoch=False, logger=True, sync_dist=True, on_step=True)
        
        optimizer.step()
        aux_optimizer.step()

    def on_validation_start(self):
        for model_name, model_instance in self.model_wrapper.items():
            model_instance.update()


    def validation_step(self, batch, batch_idx):
        for lmbda, (model_name, model_instance) in zip(self.lmbda, self.model_wrapper.items()):
            model_instance: PFrameCodec

            input_frames_dict, ref_frame_dict, ref_bpp_dict, ref_psnr_dict, ref_msssim_dict = batch

            out_compress = model_instance.compress(
                PFrameCodecCompressInput(
                    I_frame=ref_frame_dict[lmbda],
                    P_frames=input_frames_dict[lmbda],
                    I_frame_bpp=ref_bpp_dict[lmbda],
                    I_frame_psnr=ref_psnr_dict[lmbda],
                    I_frame_ms_ssim=ref_msssim_dict[lmbda]
                )
            )
            
            self.log_dict({
                f"val/{model_name}.bpp": out_compress.bpp,
                f"val/{model_name}.psnr": out_compress.psnr,
                f"val/{model_name}.ms-ssim": out_compress.ms_ssim,
            }, on_step=False, on_epoch=True, logger=True, sync_dist=True)


    def on_test_start(self):
        self.metric = MetricLogger(save_dir=self.trainer.default_root_dir)
        for model_name, model_instance in self.model_wrapper.items():
            model_instance.update()

    def test_step(self, batch, batch_idx):
        for lmbda, (model_name, model_instance) in zip(self.lmbda, self.model_wrapper.items()):
            model_instance: Union[PFrameCodec, CompressionModel]
            input_frames_dict, ref_frame_dict, ref_bpp_dict, ref_psnr_dict, ref_msssim_dict = batch
            

            with self.metric.timer(model_name, "time_compress") as timer:
                out_compress = model_instance.compress(
                    PFrameCodecCompressInput(
                        I_frame=ref_frame_dict[lmbda],
                        P_frames=input_frames_dict[lmbda],
                        I_frame_bpp=ref_bpp_dict[lmbda],
                        I_frame_psnr=ref_psnr_dict[lmbda],
                        I_frame_ms_ssim=ref_msssim_dict[lmbda]
                    )
                )


            with self.metric.timer(model_name, "time_decompress") as timer:
                model_instance.decompress(out_compress)


            self.metric.log(model_name,{
                "bpp": out_compress.bpp,
                "psnr": out_compress.psnr,
                "ms-ssim": out_compress.ms_ssim,
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
        return optimizer, aux_optimizer


class IPFrameCodecLightningModule(L.LightningModule):
    """
    In CompressAI implementation, the models contain a image codec to compress the I frame, 
    and a video frame codec to compress the P frames, which is different from the training stategy of DVC.
    """
    def __init__(self, 
                 model: IPFrameCodec,
                 ext_params: Dict[str, Any] = {}):
        super().__init__()

        # As there are two optimizers, we need to set the automatic optimization to False
        self.automatic_optimization = False

        self.lmbda = ext_params.get("lmbda", 1)
        self.lr = ext_params.get("lr", 1e-4)

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

        for lmbda, (model_name, model_instance) in zip(self.lmbda, self.model_wrapper.items()):
            model_instance: Union[IPFrameCodec, CompressionModel]
            out = model_instance.forward(
                IPFrameCodecForwardInput(
                    I_frame=batch[0],
                    P_frames=batch[1]
                )
            )

            loss = lmbda * 255 ** 2 * out.mse_loss + out.bpp
            # aux_loss_sum = sum(aux_loss for aux_loss in model_instance.aux_loss())
            aux_loss = model_instance.aux_loss()

            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(model_instance.parameters(), 1.0)
            self.manual_backward(aux_loss)

            # show metrics of the first model on the progress bar
            if model_name == "codec_0" or model_name == "codec":
                self.log_dict({
                    "loss": loss, 
                    "bpp": out.bpp,
                    "psnr": out.psnr
                }, prog_bar=True, on_step=True, on_epoch=False, logger=False)
        
            self.log_dict({
                f"train/{model_name}.loss": loss,
                f"train/{model_name}.bpp": out.bpp,
                f"train/{model_name}.psnr": out.psnr
            }, on_epoch=False, logger=True, sync_dist=True, on_step=True)

        optimizer.step()
        aux_optimizer.step()

    def on_validation_start(self):
        for model_name, model_instance in self.model_wrapper.items():
            model_instance.update()

    def validation_step(self, batch, batch_idx):
        for model_name, model_instance in self.model_wrapper.items():
            model_instance: Union[IPFrameCodec, CompressionModel]
            out_compress = model_instance.compress(
                IPFrameCodecCompressInput(
                    I_frame=batch[0],
                    P_frames=batch[1]
                )
            )

            self.log_dict({
                f"val/{model_name}.bpp": out_compress.bpp,
                f"val/{model_name}.psnr": out_compress.psnr,
                f"val/{model_name}.ms-ssim": out_compress.ms_ssim,
            }, on_step=False, on_epoch=True, logger=True, sync_dist=True)

    def on_test_start(self):
        self.metric = MetricLogger(save_dir=self.trainer.default_root_dir)
        for model_name, model_instance in self.model_wrapper.items():
            model_instance.update()

    def test_step(self, batch, batch_idx):
        for model_name, model_instance in self.model_wrapper.items():
            model_instance: Union[IPFrameCodec, CompressionModel]

            with self.metric.timer(model_name, "time_compress") as timer:
                out_compress = model_instance.compress(
                    IPFrameCodecCompressInput(
                        I_frame=batch[0],
                        P_frames=batch[1]
                    )
                )

            with self.metric.timer(model_name, "time_decompress") as timer:
                model_instance.decompress(out_compress)
                
            self.metric.log(model_name,{
                "bpp": out_compress.bpp,
                "psnr": out_compress.psnr,
                "ms-ssim": out_compress.ms_ssim,
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
        

# FIXME: DVC uses BitEstimator to estimate the bpp during the training stage,
# but it is not compatible with the CompressAI entropy bottleneck and there is a
# performance degradation when setting the same lambda for CompressAI entropy bottleneck. 
# So further investigation is needed.
# class DVCTrainer(L.LightningModule):
#     def __init__(self,
#                  model: nn.Module,
#                  ext_params: Dict[str, Any] = {}
#                  ):
#         super().__init__()

#         self.automatic_optimization = False

#         if "lmbda" not in ext_params.keys():
#             raise ValueError("lmbda is required in ext_params")

#         self.lmbda = ext_params.get("lmbda")
#         self.lr = ext_params.get("lr", 1e-4)
#         self.lr_decay = ext_params.get("lr_decay", 0.1)
#         self.lr_decay_interval = ext_params.get("lr_decay_interval", 1_800_000)

#         self.model_wrapper: nn.ModuleDict[str, CompressionModel] = nn.ModuleDict({})

#         if isinstance(self.lmbda, list):
#             for idx in range(len(self.lmbda)):
#                 self.model_wrapper[f"codec_{idx}"] = deepcopy(model)
#             del model
#         else:
#             self.lmbda = [self.lmbda]
#             self.model_wrapper["codec"] = model
    
#     def on_train_batch_start(self, batch, batch_idx):
#         if self.global_step == self.lr_decay_interval:
#             for optimizer in self.optimizers():
#                 for param_group in optimizer.param_groups:
#                     param_group["lr"] *= self.lr_decay

#     def training_step(self, batch, batch_idx):
#         optimizer = self.optimizers()
#         optimizer.zero_grad()

#         input_frame, ref_frame = batch
#         N, _, H, W = input_frame.shape

#         for lmbda, (model_name, model_instance) in zip(self.lmbda, self.model_wrapper.items()):
#             out = model_instance.forward(input_frame, ref_frame)
            
#             bpp_loss = torch.mean(out["bpp"]["bpp"])
#             bpp_feature = torch.mean(out["bpp"]["bpp_feature"])
#             bpp_z = torch.mean(out["bpp"]["bpp_z"])
#             bpp_mv = torch.mean(out["bpp"]["bpp_mv"])
            
#             mse_loss = torch.mean(out["mse_loss"])
#             warploss = torch.mean(out["warploss"])
#             interloss = torch.mean(out["interloss"])
            
#             psnr = 10 * torch.log10(1. / mse_loss)
#             warp_psnr = 10 * torch.log10(1. / warploss)
#             inter_psnr = 10 * torch.log10(1. / interloss)

#             #TODO
#             if self.global_step < 500000:
#                 warp_weight = 0.1
#             else:
#                 warp_weight = 0
#             distortion_loss = mse_loss + warp_weight * (warploss + interloss)

#             loss = lmbda * distortion_loss + bpp_loss
#             self.manual_backward(loss)
#             torch.nn.utils.clip_grad_norm_(model_instance.parameters(), 0.5)

#             if model_name == "codec_0" or model_name == "codec":
#                 self.log_dict(dict(
#                     loss=loss,
#                     bpp=bpp_loss,
#                     bpp_mv=bpp_mv,
#                     bpp_feature=bpp_feature,
#                     bpp_z=bpp_z,
#                     psnr=psnr,
#                 ), prog_bar=True, on_step=True, on_epoch=False, logger=False)

#             self.log_dict({
#                 f"train/{model_name}.loss": loss,
#                 f"train/{model_name}.bpp": bpp_loss,
#                 f"train/{model_name}.bpp_mv": bpp_mv,
#                 f"train/{model_name}.bpp_feature": bpp_feature,
#                 f"train/{model_name}.bpp_z": bpp_z,
#                 f"train/{model_name}.psnr": psnr,
#                 f"train/{model_name}.warp_psnr": warp_psnr,
#                 f"train/{model_name}.inter_psnr": inter_psnr,
#             }, on_epoch=False, logger=True, sync_dist=True, on_step=True)
        
#         optimizer.step()

#     def validation_step(self, batch, batch_idx):
#         for lmbda, (model_name, model_instance) in zip(self.lmbda, self.model_wrapper.items()):

#             input_frames_dict, ref_frame_dict, ref_bpp_dict, ref_psnr_dict, ref_msssim_dict = batch

#             input_frames = input_frames_dict[lmbda]
#             ref_frame = ref_frame_dict[lmbda].squeeze(0)
#             seqlen = input_frames.shape[1]

#             avg_bpp = ref_bpp_dict[lmbda]
#             avg_psnr = ref_psnr_dict[lmbda]
#             avg_msssim = ref_msssim_dict[lmbda]

#             for i in range(seqlen):
#                 input_frame = input_frames[:, i, :, :, :]
#                 out = model_instance.forward(input_frame, ref_frame)
                
#                 bpp = torch.mean(out["bpp"]["bpp"])
#                 mse_loss = torch.mean(out["mse_loss"])
#                 psnr = 10 * torch.log10(1. / mse_loss)
#                 ms_ssim_loss = ms_ssim(out["recon_frame"], input_frame, data_range=1)
#                 avg_bpp += bpp
#                 avg_psnr += psnr
#                 avg_msssim += ms_ssim_loss
                
#                 ref_frame = out["recon_frame"]

#             self.log_dict({
#                 f"val/{model_name}.bpp": avg_bpp / (seqlen + 1),
#                 f"val/{model_name}.psnr": avg_psnr / (seqlen + 1),
#                 f"val/{model_name}.ms-ssim": avg_msssim / (seqlen + 1),
#             }, on_step=False, on_epoch=True, logger=True, sync_dist=True)

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(
#             [p for model_instance in self.model_wrapper.values() for p in model_instance.parameters()],
#             lr=self.lr
#         )
#         return optimizer