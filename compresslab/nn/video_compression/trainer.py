import lightning as L
from compresslab.core.models import CompressionModel
from compresslab.utils.logger import MetricLogger
from compresslab.nn.video_compression.abc import *
import torch
import torch.nn as nn
from copy import deepcopy
from typing import Dict, Any, Union
from torch import vmap
from torch.func import stack_module_state, functional_call


#TODO: vmap support
class ModelWrapper(nn.Module):
    def __init__(self, lmbda: List[float], model: VideoCodec, codec):
        super().__init__()

        self.lmbda = lmbda if isinstance(lmbda, list) else [lmbda]
        self.models = nn.ModuleList([deepcopy(model) for _ in range(len(lmbda))])

        for model in self.models:
            model.preprocess(codec)

        def fmodel(params, buffers, x):
            return functional_call(model, (params, buffers), (x,))
        
        self.vmap_forward = vmap(fmodel, in_dims=(0, 0, None), randomness="different")

    def forward(self, input) -> List[VideoCodecForwardOutput]:
        self.param, self.buffer = stack_module_state(self.models)
        vmap_output = self.vmap_forward(self.param, self.buffer, input)
        
        return vmap_output

    def aux_loss(self) -> torch.Tensor:
        aux_loss = 0.0
        for model in self.models:
            aux_loss += model.aux_loss()
        return aux_loss

    def items(self):
        return {f"codec_{idx}": model for idx, model in enumerate(self.models)}.items()


class VideoCodecTrainer(L.LightningModule):
    def __init__(self, 
                 model: VideoCodec,
                 ext_params: Dict[str, Any] = {}
                 ):
        super().__init__()

        self.automatic_optimization = False


        self.enable_vmap = False

        self.lmbda = ext_params.get("lmbda", 1)
        self.lr = ext_params.get("lr", 1e-4)
        codec = ext_params.get("codec", None)

        self.model_wrapper = ModelWrapper(self.lmbda, model, codec)

    # TODO: multi-stage training

    def training_step(self, batch, batch_idx):
        optimizer, aux_optimizer = self.optimizers()
        optimizer.zero_grad()
        aux_optimizer.zero_grad()


        if self.enable_vmap:
            out = self.model_wrapper.forward(batch)

            total_loss = 0.0
            for idx, lmbda in enumerate(self.lmbda):
                loss = lmbda * 255 ** 2 * out["mse_loss"][idx] + out["bpp"][idx]
                
                if idx == 0:
                    self.log_dict({
                        "loss": loss, 
                        "bpp": out["bpp"][idx],
                        "psnr": out["psnr"][idx]
                    }, prog_bar=True, on_step=True, on_epoch=False, logger=False)
                
                self.log_dict({
                    f"train/codec_{idx}.loss": loss,
                    f"train/codec_{idx}.bpp": out["bpp"][idx],
                    f"train/codec_{idx}.psnr": out["psnr"][idx]
                }, on_epoch=False, logger=True, sync_dist=True, on_step=True)

                total_loss += loss

            total_aux_loss = self.model_wrapper.aux_loss()

        else:
            total_loss = 0.0
            total_aux_loss = 0.0

            for lmbda, (model_name, model_instance) in zip(self.lmbda, self.model_wrapper.items()):
                model_instance: Union[VideoCodec, CompressionModel]

                out = model_instance.forward_all(
                    VideoCodecForwardInput(
                        frames=batch
                    )
                )

                loss = lmbda * 255 ** 2 * out.mse_loss + out.bpp
                aux_loss = model_instance.aux_loss()

                if model_name == "codec_0":
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
                
                total_loss += loss
                total_aux_loss += aux_loss

        self.manual_backward(total_loss)
        torch.nn.utils.clip_grad_norm_(self.model_wrapper.parameters(), 1.0)
        self.manual_backward(total_aux_loss)

        optimizer.step()
        aux_optimizer.step()


    # def validation_step(self, batch, batch_idx):
    #     for model_name, model_instance in self.model_wrapper.items():
    #         model_instance: Union[VideoCodec, CompressionModel]

    #         out = model_instance.forward(
    #             VideoCodecForwardInput(
    #                 frames=batch
    #             )
    #         )

    #         self.log_dict({
    #             f"val/{model_name}.bpp": out.bpp,
    #             f"val/{model_name}.psnr": out.psnr,
    #         }, on_step=False, on_epoch=True, logger=True, sync_dist=True)


    def on_test_start(self):
        self.metric = MetricLogger(save_dir=self.trainer.default_root_dir)
        for model_name, model_instance in self.model_wrapper.items():
            model_instance.update()

    def test_step(self, batch, batch_idx):
        for model_name, model_instance in self.model_wrapper.items():
            model_instance: Union[VideoCodec, CompressionModel]

            with self.metric.timer(model_name, "time_compress") as timer:
                out_compress = model_instance.compress(
                    VideoCodecCompressInput(
                        frames=batch
                    )
                )

            with self.metric.timer(model_name, "time_decompress") as timer:
                out_decompress = model_instance.decompress(out_compress)
                
            self.metric.log(model_name, {
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