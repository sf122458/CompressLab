# import lightning as L
# from compresslab.core.models import CompressionModel
# from compresslab.utils.logger import MetricsLogger
# from compresslab.nn.video_compression.abc import *
# import torch
# from typing import Dict, Any, Union, Type
# from compresslab.nn.base import ModelWrapper
# from compresslab.nn.base import BasicTrainer


# # TODO: modify `VideoCodecTrainer` to compatible with `CompressAICodecTrainer`
# class VideoCodecTrainer(BasicTrainer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         self.automatic_optimization = False

#         assert self.training_mode in ["fast", "medium", "slow"]

        

#         self.model_wrapper = ModelWrapper(model_class, params, len(self.lmbda))

#         for model_instance in self.model_wrapper.values():
#             model_instance: Union[VideoCodec, CompressionModel]
#             model_instance.preprocess(codec)


#     # TODO: multi-stage training

#     def training_step(self, batch, batch_idx):
#         optimizer, aux_optimizer = self.optimizers()
#         optimizer.zero_grad()
#         aux_optimizer.zero_grad()


#         if self.training_mode == "fast":
#             out = self.model_wrapper.forward(batch)

#             total_loss = 0.0
#             for idx, lmbda in enumerate(self.lmbda):
#                 loss = lmbda * 255 ** 2 * out["mse_loss"][idx] + out["bpp"][idx]
                
#                 if idx == 0:
#                     self.log_dict({
#                         "loss": loss, 
#                         "bpp": out["bpp"][idx],
#                         "psnr": out["psnr"][idx]
#                     }, prog_bar=True, on_step=True, on_epoch=False, logger=False)
                
#                 self.log_dict({
#                     f"train/codec_{idx}.loss": loss,
#                     f"train/codec_{idx}.bpp": out["bpp"][idx],
#                     f"train/codec_{idx}.psnr": out["psnr"][idx]
#                 }, on_epoch=False, logger=True, sync_dist=True, on_step=True)

#                 total_loss += loss

#             total_aux_loss = self.model_wrapper.aux_loss()

#             self.manual_backward(total_loss)
#             torch.nn.utils.clip_grad_norm_(self.model_wrapper.parameters(), 1.0)
#             self.manual_backward(total_aux_loss)

#         else:
#             total_loss = 0.0
#             total_aux_loss = 0.0

#             for lmbda, (model_name, model_instance) in zip(self.lmbda, self.model_wrapper.items()):
#                 model_instance: Union[VideoCodec, CompressionModel]

#                 out = model_instance.forward(
#                     VideoCodecForwardInput(
#                         frames=batch
#                     )
#                 )

#                 loss = lmbda * 255 ** 2 * out.mse_loss + out.bpp
#                 aux_loss = model_instance.aux_loss()

#                 if self.training_mode == "slow":
#                     self.manual_backward(loss)
#                     torch.nn.utils.clip_grad_norm_(model_instance.parameters(), 1.0)
#                     self.manual_backward(aux_loss)
#                 else:
#                     total_loss += loss
#                     total_aux_loss += aux_loss

#                 if model_name == "codec_0":
#                     self.log_dict({
#                         "loss": loss, 
#                         "bpp": out.bpp,
#                         "psnr": out.psnr
#                     }, prog_bar=True, on_step=True, on_epoch=False, logger=False)
                
#                 self.log_dict({
#                     f"train/{model_name}.loss": loss,
#                     f"train/{model_name}.bpp": out.bpp,
#                     f"train/{model_name}.psnr": out.psnr
#                 }, on_epoch=False, logger=True, sync_dist=True, on_step=True)

#             if self.training_mode == "medium":
#                 self.manual_backward(total_loss)
#                 torch.nn.utils.clip_grad_norm_(self.model_wrapper.parameters(), 1.0)
#                 self.manual_backward(total_aux_loss)

#         optimizer.step()
#         aux_optimizer.step()


#     def validation_step(self, batch, batch_idx):
#         for model_name, model_instance in self.model_wrapper.items():
#             model_instance: Union[VideoCodec, CompressionModel]

#             out = model_instance.forward(
#                 VideoCodecForwardInput(
#                     frames=batch
#                 )
#             )

#             self.log_dict({
#                 f"val/{model_name}.bpp": out.bpp,
#                 f"val/{model_name}.psnr": out.psnr,
#             }, on_step=False, on_epoch=True, logger=True, sync_dist=True)


#     def on_test_start(self):
#         self.metric = MetricsLogger(save_dir=self.trainer.default_root_dir)
#         for model_name, model_instance in self.model_wrapper.items():
#             model_instance.update()

#     def test_step(self, batch, batch_idx):
#         for model_name, model_instance in self.model_wrapper.items():
#             model_instance: Union[VideoCodec, CompressionModel]

#             with self.metric.timer(model_name, "time_compress") as timer:
#                 out_compress = model_instance.compress(
#                     VideoCodecCompressInput(
#                         frames=batch
#                     )
#                 )

#             with self.metric.timer(model_name, "time_decompress") as timer:
#                 out_decompress = model_instance.decompress(out_compress)
                
#             self.metric.log(model_name, {
#                 "bpp": out_compress.bpp,
#                 "psnr": out_compress.psnr,
#                 "ms-ssim": out_compress.ms_ssim,
#             })
#     def on_test_end(self):
#         self.metric.save()


#     def configure_optimizers(self):
#         parameters = []
#         aux_parameters = []
#         for model_name, model_instance in self.model_wrapper.items():
#             parameters += [p for n, p in model_instance.named_parameters() if p.requires_grad and not n.endswith(".quantiles")]
#             aux_parameters += [p for n, p in model_instance.named_parameters() if p.requires_grad and n.endswith(".quantiles")]
#         optimizer = torch.optim.Adam(parameters, lr=self.lr)
#         aux_optimizer = torch.optim.Adam(aux_parameters, lr=1e-3)
#         return optimizer, aux_optimizer