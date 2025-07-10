import os, logging
from compresslab.core.models import CompressionModel
from compresslab.nn.lossy_image_compression.abc import (
    ImageCodecForwardInput,
    ImageCodecForwardOutput,
    ImageCodecCompressInput,
    ImageCodec
)
import torch
from typing import Union
from pytorch_msssim import ms_ssim
from compresslab.nn.base import BaseTrainer

class ImageCodecTrainer(BaseTrainer):
    def loss_fn(self, lmbda, out: ImageCodecForwardOutput):
        if self.global_step < self.key_step["fine_tune"]:
            return lmbda * out.mse_loss * 255 ** 2 + out.bpp
        else:
            return lmbda * out.ms_ssim_loss + out.bpp
    
    def on_train_batch_end(self, output, batch, batch_idx):
        if self.global_step == self.key_step["lr_decay"]:
            optimizer = self.optimizers()
            optimizer.param_groups[0]["lr"] *= 0.1
            logging.info(f"Learning rate decayed to {optimizer.param_groups[0]['lr']} at step {self.global_step}")
        
        if self.global_step == self.key_step["fine_tune"]:
            self.model_wrapper.update()
            self.trainer.save_checkpoint(os.path.join(self.trainer.default_root_dir, f"checkpoints/mse.ckpt"), weights_only=True)
            logging.info(f"Saving checkpoint trained on `MSE` at step {self.global_step}.")

        if self.global_step == self.trainer.max_steps and self.key_step["fine_tune"] != self.trainer.max_steps:
            self.model_wrapper.update()
            self.trainer.save_checkpoint(os.path.join(self.trainer.default_root_dir, f"checkpoints/ms-ssim.ckpt"), weights_only=True)
            logging.info(f"Saving checkpoint fine-tuned on `MS-SSIM` at step {self.global_step}.")
    

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()

        total_loss = 0.0
        total_aux_loss = 0.0

        for lmbda, (model_name, model_instance) in zip(self.lmbda, self.model_wrapper.items()):
            model_instance: Union[ImageCodec, CompressionModel]

            out = model_instance.forward(
                ImageCodecForwardInput(
                    x=batch
                )
            )

            loss = self.loss_fn(lmbda, out)

            total_loss += loss
            total_aux_loss += model_instance.aux_loss()

            # show metrics of the first model on the progress bar
            if model_name == "codec_0":
                self.bar_metrics({
                    "loss": loss,
                    "bpp": out.bpp,
                    "psnr": out.psnr,
                    "ms-ssim": out.ms_ssim
                })

            self.log_train_metrics(model_name, {
                "loss": loss,
                "bpp": out.bpp,
                "psnr": out.psnr,
                "ms-ssim": out.ms_ssim
            })

        self.manual_backward(total_loss)
        torch.nn.utils.clip_grad_norm_(self.model_wrapper.parameters(), 1.0)
        self.manual_backward(total_aux_loss)

        self.log_train_monitor({
            "lr": optimizer.param_groups[0]["lr"],
        })

        optimizer.step()

    def validation_step(self, batch, batch_idx):
        for model_name, model_instance in self.model_wrapper.items():
            model_instance: Union[ImageCodec, CompressionModel]
            out = model_instance.forward(
                ImageCodecForwardInput(
                    x=batch
                )
            )

            self.log_val_metrics(model_name, {
                "bpp": out.bpp,
                "psnr": out.psnr,
                "ms-ssim": out.ms_ssim
            })

    def test_step(self, batch, batch_idx):
        for model_name, model_instance in self.model_wrapper.items():
            model_instance: Union[ImageCodec, CompressionModel]
            with self.metric.timer(model_name, "encoding_time"):
                out_compress = model_instance.compress(
                    ImageCodecCompressInput(
                        x=batch
                    )
                )
            with self.metric.timer(model_name, "decoding_time"):
                out_decompress = model_instance.decompress(out_compress)

            mse_loss = torch.nn.functional.mse_loss(out_decompress.x_hat, batch)

            psnr = 10 * torch.log10(1 / mse_loss)

            ms_ssim_metric = ms_ssim(
                out_decompress.x_hat, 
                batch, 
                data_range=1.0, 
                size_average=True
            )

            self.log_test_metrics(model_name, {
                "bpp": out_compress.bpp,
                "psnr": psnr,
                "ms-ssim": ms_ssim_metric
            })