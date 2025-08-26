import os, logging
from compresslab.core.models import CompressionModel
import torch
from compresslab.nn.base import CompressAIImageCodecTrainer
from compresslab.nn.base.metrics import ImageMetricsOutput
from torchvision.utils import save_image

class ImageCodecTrainer(CompressAIImageCodecTrainer):
    def loss_fn(self, lmbda, out: ImageMetricsOutput):
        if self.global_step < self.finetune_step:
            return lmbda * out.mse_loss * 255 ** 2 + out.bpp
        else:
            return lmbda * out.ms_ssim_loss + out.bpp
    
    # def on_train_batch_end(self, output, batch, batch_idx):
    #     if self.global_step == self.key_step["lr_decay"]:
    #         optimizer = self.optimizers()
    #         optimizer.param_groups[0]["lr"] *= 0.1
    #         logging.info(f"Learning rate decayed to {optimizer.param_groups[0]['lr']} at step {self.global_step}")

    def on_train_batch_end(self, output, batch, batch_idx):
        if self.global_step == self.finetune_step - 1:
            self.trainer.save_checkpoint(
                os.path.join(self.trainer.default_root_dir, f"checkpoints/mse.ckpt"), 
                weights_only=True
            )
            logging.info(f"Saving checkpoint trained on `MSE` at step {self.global_step}.")
        
        if self.global_step == self.trainer.max_steps and self.finetune_step < self.trainer.max_steps:
            self.trainer.save_checkpoint(
                os.path.join(self.trainer.default_root_dir, f"checkpoints/ms_ssim.ckpt"), 
                weights_only=True
            )
            logging.info(f"Saving checkpoint fine-tuned on `MS-SSIM` at step {self.global_step}.")

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        total_loss = 0.0
        total_aux_loss = 0.0

        for lmbda, (model_name, model_instance) in zip(self.lmbda, self.model_wrapper.items()):
            model_instance: CompressionModel

            out = model_instance(batch)

            metrics = self.metrics_collector.forward(batch, out["x_hat"], 
                                                     likelihoods=out["likelihoods"],
                                                     mse=True, ms_ssim=True)
            loss = self.loss_fn(lmbda, metrics)

            total_loss += loss
            total_aux_loss += model_instance.aux_loss()

            # show metrics of the first model on the progress bar
            if model_name == "codec_0":
                self.bar_metrics({
                    "loss": loss,
                    "bpp": metrics.bpp,
                    "psnr": metrics.psnr,
                    "ms-ssim": metrics.ms_ssim
                })

            self.log_train_metrics(model_name, {
                "loss": loss,
                "bpp": metrics.bpp,
                "psnr": metrics.psnr,
                "ms-ssim": metrics.ms_ssim
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
            model_instance: CompressionModel
            out = model_instance(batch)
            
            metrics = self.metrics_collector.forward(batch, out["x_hat"], 
                                                     likelihoods=out["likelihoods"],
                                                     mse=True, ms_ssim=True)

            self.log_val_metrics(model_name, {
                "bpp": metrics.bpp,
                "psnr": metrics.psnr,
                "ms-ssim": metrics.ms_ssim
            })

    def test_step(self, batch, batch_idx):
        x, filename = batch["image"], batch["filename"][0]
        for model_name, model_instance in self.model_wrapper.items():
            model_instance: CompressionModel
            with self.metrics_logger.timer(model_name, "compress"):
                out_compress = model_instance.compress(x)
                
                if self.ext_params.SaveBitstream:
                    self.write_bitstream(f"{model_name}/{filename}", **out_compress)

            with self.metrics_logger.timer(model_name, "compress"):
                if self.ext_params.SaveBitstream:
                    out_decompress = self.read_bitstream(f"{model_name}/{filename}")

                out_decompress = model_instance.decompress(**out_compress)
                
            metrics = self.metrics_collector.forward(x, out_decompress["x_hat"], 
                                                     strings=out_compress["strings"],
                                                     mse=True, ms_ssim=True)
            self.log_test_metrics(model_name, {
                "bpp": metrics.bpp,
                "psnr": metrics.psnr,
                "ms-ssim": metrics.ms_ssim
            })

            if self.ext_params.SaveRecon:
                self.save_recon_imgs(
                    out_decompress["x_hat"], 
                    f"{model_name}/{filename}_{self.model_type}_{metrics.bpp:.4f}_{metrics.psnr:.2f}_{metrics.ms_ssim:.4f}.png"
                )