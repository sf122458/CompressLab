"""
StableCodec
Modified based on https://github.com/LuizScarlet/StableCodec
Paper: https://arxiv.org/abs/2506.21977
"""
import numpy as np
import torch
import torch.nn as nn
import logging
from peft import LoraConfig
from diffusers import (
    AutoencoderKL, 
    UNet2DConditionModel, 
    DDPMScheduler
)
from transformers import CLIPTextModel, AutoTokenizer
from compresslab.nn.base import BasicTrainer
from compresslab.nn.lossy_image_compression.elic.models import ELIC
from compresslab.nn.generative_image_compression.StableCodec.utils.tile import VAEHook
from compresslab.nn.generative_image_compression.StableCodec.utils.lora import lora_fwd
from compresslab.nn.generative_image_compression.StableCodec.latent_codec import LatentCodec
from compresslab.utils.constant import PRETRAINED_CACHE_DIR

class StableCodec(BasicTrainer):
    
    pretrained_level = [1, 2, 3, 4, 5]
    level_ckpt_slot = {l: str(2**l) for l in pretrained_level}
    
    def __init__(
        self,
        level: int = None,
        elic_path: str = f"{PRETRAINED_CACHE_DIR}/StableCodec/elic_official.pth",
        sd_path = "stabilityai/sd-turbo",
        latent_tiled_size: int = 96,
        latent_tiled_overlap: int = 32, 
        vae_encoder_tiled_size: int = 1024,
        vae_decoder_tiled_size: int = 160,
        lora_rank_vae: int = 16,
        lora_rank_unet: int = 32,
        pos_prompt: str = "A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.",
        color_fix: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        assert level in self.pretrained_level or level is None, f"level should be in {self.pretrained_level} or None, but got {level}."

        self.sd_path = sd_path
        self.latent_tiled_size = latent_tiled_size
        self.latent_tiled_overlap = latent_tiled_overlap

        
        self.tokenizer = AutoTokenizer.from_pretrained(
            sd_path,
            subfolder="tokenizer",
            cache_dir=PRETRAINED_CACHE_DIR,
        )
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            sd_path,
            subfolder="text_encoder",
            cache_dir=PRETRAINED_CACHE_DIR,
        )
        
        self.text_encoder.requires_grad_(False)
        
        self.vae = AutoencoderKL.from_pretrained(
            sd_path,
            subfolder="vae",
            cache_dir=PRETRAINED_CACHE_DIR,
        )
        
        self.unet = UNet2DConditionModel.from_pretrained(
            sd_path,
            subfolder="unet",
            cache_dir=PRETRAINED_CACHE_DIR,
        )
        
        self.register_buffer("timesteps", torch.tensor([999]).long())

        self._init_tiled_vae(
            encoder_tile_size=vae_encoder_tiled_size,
            decoder_tile_size=vae_decoder_tiled_size,
            color_fix=color_fix,
        )
        
        # LoRA configurations
        target_modules_vae = r"^encoder\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\.0)$"
        target_modules_unet = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
            "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
        ]
        
        vae_lora_config = LoraConfig(
            r=lora_rank_vae, 
            init_lora_weights="gaussian",
            target_modules=target_modules_vae,
        )
        self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        unet_lora_config = LoraConfig(
            r=lora_rank_unet,
            init_lora_weights="gaussian",
            target_modules=target_modules_unet,
        )
        self.unet.add_adapter(unet_lora_config)

        self.vae_lora_layers = []
        for name, module in self.vae.named_modules():
            if "base_layer" in name:
                self.vae_lora_layers.append(name[:-len(".base_layer")])
        for name, module in self.vae.named_modules():
            if name in self.vae_lora_layers:
                module.forward = lora_fwd.__get__(module, module.__class__)

        self.unet_lora_layers = []
        for name, module in self.unet.named_modules():
            if "base_layer" in name:
                self.unet_lora_layers.append(name[:-len(".base_layer")])
        for name, module in self.unet.named_modules():
            if name in self.unet_lora_layers:
                module.forward = lora_fwd.__get__(module, module.__class__)

        # load latent codec
        self.codec = LatentCodec()
        self.unet.conv_in = nn.Conv2d(
            in_channels=320,
            out_channels=320,
            kernel_size=3,
            stride=1,
            padding=1,
        ).to(self.device)

        # process prompt
        self._set_prompt(pos_prompt)
        
        # Auxiliary models
        aux_model = ELIC(
            N=192,
            M=320,
            slice_num=10,
            slice_ch=[8, 8, 8, 8, 16, 16, 32, 32, 96, 96],
        )
        aux_model.load_state_dict(torch.load(elic_path))
        self.aux_codec = aux_model.g_a
        self.aux_codec.eval()
        self.aux_codec.requires_grad_(False)
        
        # load ckpt
        if level is not None:
            codec_path = f"{PRETRAINED_CACHE_DIR}/StableCodec/stablecodec_ft{self.level_ckpt_slot[level]}.pkl"
            logging.info(f"Loading pretrained weights from {codec_path}")
            ckpt = torch.load(codec_path, map_location="cpu")
            _sd_codec = self.codec.state_dict()
            for k in ckpt["state_dict_codec"]:
                _sd_codec[k] = ckpt["state_dict_codec"][k]
            self.codec.load_state_dict(_sd_codec)
            
            _sd_vae = self.vae.state_dict()
            for k in ckpt["state_dict_vae"]:
                _sd_vae[k] = ckpt["state_dict_vae"][k]
            self.vae.load_state_dict(_sd_vae)

            _sd_unet = self.unet.state_dict()
            for k in ckpt["state_dict_unet"]:
                _sd_unet[k] = ckpt["state_dict_unet"][k]
            self.unet.load_state_dict(_sd_unet)
        else:
            logging.info(f"Training from scratch.")
            
    def compress(self, imgs):
        """Compress images.

        Args:
            imgs (torch.Tensor): Target images with shape (N, 3, H, W) in [0, 1].
        """
        aux_latent = self.aux_codec(imgs)
        lq_latent = self.vae.encode(imgs*2-1).latent_dist.mode() * self.vae.config.scaling_factor
        return self.codec.compress(lq_latent, aux_latent)

    def decompress(self, strings, shape, pos_prompt=[1]):
        lq_latent_hat, res = self.codec.decompress(strings, shape)
        pos_caption_enc = [self.pos_caption_enc for i in range(len(pos_prompt))]
        pos_caption_enc = torch.cat(pos_caption_enc, dim=0).to(lq_latent_hat.device)

        # One-step denoiser with tile function
        _, _, h, w = lq_latent_hat.shape
        tile_size, tile_overlap = self.latent_tiled_size, self.latent_tiled_overlap
        if h * w <= tile_size ** 2:
            model_pred = self.unet(
                lq_latent_hat, 
                self.timesteps,
                encoder_hidden_states=pos_caption_enc
            ).sample
        else:
            logging.info(f"The input latent shape is ({h}, {w}), needs to tiled.")
            tile_size = min(tile_size, min(h, w))
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1).to(lq_latent_hat.device)

            grid_rows = 0
            cur_x = 0
            while cur_x < lq_latent_hat.size(-1):
                cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
                grid_rows += 1
                
            grid_cols = 0
            cur_y = 0
            while cur_y < lq_latent_hat.size(-2):
                cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
                grid_cols += 1

            input_list = []
            noise_preds = []
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols - 1 or row < grid_rows - 1:
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    input_tile = lq_latent_hat[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    input_list.append(input_tile)

                    if len(input_list) == 1 or col == grid_cols-1:
                        input_list_t = torch.cat(input_list, dim=0)
                        model_pred = self.unet(input_list_t, self.timesteps, encoder_hidden_states=pos_caption_enc).sample
                        input_list = []
                    noise_preds.append(model_pred)

            noise_pred = torch.zeros(lq_latent_hat[:, :4].shape, device=lq_latent_hat.device)
            contributors = torch.zeros(lq_latent_hat[:, :4].shape, device=lq_latent_hat.device)
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
            noise_pred /= contributors
            model_pred = noise_pred

        x_denoised = self.scheduler.step(model_pred, self.timesteps, lq_latent_hat[:, :4], return_dict=True).prev_sample + res

        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        output_image = (output_image + 1) / 2
        return output_image
    
    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return super().validation_step(batch, batch_idx)

    def on_test_start(self):
        super().on_test_start()
        self.codec.update()
        self.scheduler = self._make_one_step_scheduler(self.sd_path)
    
    def test_step(self, batch, batch_idx):
        imgs, filename = batch["image"], batch["filename"][0]
        
        with self.metrics_logger.timer("StableCodec", "compress"):
            out_compress = self.compress(imgs)
            if self.ext_params.SaveBitstream:
                self.write_bitstream(filename, **out_compress)
                
        with self.metrics_logger.timer("StableCodec", "decompress"):
            if self.ext_params.SaveBitstream:
                out_compress = self.read_bitstream(filename)
                    
            preds = self.decompress(**out_compress)
        
        metrics = self.metrics_collector.forward(
            imgs, preds, 
            strings=out_compress["strings"],
            psnr=True, ms_ssim=True, lpips=True, dists=True, kid=True, fid=True
        )

        self.log_test_metrics(
            "StableCodec",
            {
                "bpp": metrics.bpp,
                "psnr": metrics.psnr,
                "ms-ssim": metrics.ms_ssim,
                "lpips": metrics.lpips,
                "dists": metrics.dists,
            },
            {
                "kid": metrics.kid,
                "fid": metrics.fid,
            }
        )
        
        if self.ext_params.SaveRecon:
            self.save_recon_imgs(preds, f"{filename}_{metrics.bpp:.4f}.png")
    
    def configure_optimizers(self):
        return super().configure_optimizers()

    def _make_one_step_scheduler(self, pretrained_path):
        noise_scheduler_one_step = DDPMScheduler.from_pretrained(
            pretrained_path, 
            subfolder="scheduler", 
            cache_dir=PRETRAINED_CACHE_DIR
        )
        noise_scheduler_one_step.set_timesteps(1, device=self.device)
        noise_scheduler_one_step.alphas_cumprod = noise_scheduler_one_step.alphas_cumprod.to(self.device)
        return noise_scheduler_one_step
    
    def _init_tiled_vae(
        self, 
        encoder_tile_size: int = 256,
        decoder_tile_size: int = 256,
        fast_encoder: bool = True,
        fast_decoder: bool = False,
        color_fix: bool = False,
        vae_to_gpu=True,
    ):
        encoder = self.vae.encoder
        decoder = self.vae.decoder
        
        # save original forward (only once)
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward', self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)
        
        self.vae.encoder.forward = VAEHook(
            encoder,
            tile_size=encoder_tile_size,
            is_decoder=False,
            fast_encoder=fast_encoder,
            fast_decoder=fast_decoder,
            color_fix=color_fix,
            to_gpu=vae_to_gpu,
        )
        
        self.vae.decoder.forward = VAEHook(
            decoder,
            tile_size=decoder_tile_size,
            is_decoder=True,
            fast_encoder=fast_encoder,
            fast_decoder=fast_decoder,
            color_fix=color_fix,
            to_gpu=vae_to_gpu,
        )
        
    def _set_prompt(self, pos_prompt):
        caption_tokens = self.tokenizer(
            pos_prompt, 
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(self.device)
        self.pos_caption_enc = self.text_encoder(caption_tokens)[0]
        del self.tokenizer, self.text_encoder
        
    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""

        var = 0.01
        midpoint = (tile_width - 1) / 2 # -1 because index goes from 0 to latent_width - 1
        x_probs = [np.exp(-(x-midpoint)*(x-midpoint)/(tile_width ** 2)/(2*var)) / np.sqrt(2*np.pi*var) 
                   for x in range(tile_width)]
        midpoint = tile_height / 2
        y_probs = [np.exp(-(y-midpoint)*(y-midpoint)/(tile_height ** 2)/(2*var)) / np.sqrt(2*np.pi*var) 
                   for y in range(tile_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights), (nbatches, self.unet.config.in_channels, 1, 1))

