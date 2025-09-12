"""
Modified based on https://github.com/huai-chang/DiffEIC.
Paper: [Towards Extreme Image Compression with Latent Feature Guidance and Diffusion Prior](https://arxiv.org/pdf/2404.18820)
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, Tuple, List
import logging
import os
import math
from dataclasses import dataclass
from compresslab.nn.generative_image_compression.DiffEIC.compression.lfgcm import LFGCM
from compresslab.nn.generative_image_compression.DiffEIC.reconstruction.cddm import CDDM
from compresslab.nn.generative_image_compression.DiffEIC.reconstruction.sampler import SpacedSampler, DDIMSampler
from compresslab.nn.generative_image_compression.DiffEIC.reconstruction.models import LatentDiffusion
from compresslab.nn.base import BasicTrainer
from compresslab.utils.constant import PRETRAINED_CACHE_DIR

@dataclass
class PreprocessOutput:
    z_guidance: torch.Tensor = None
    z_content: torch.Tensor = None
    bpp: torch.Tensor = None
    q_bpp: torch.Tensor = None

class DiffEIC(BasicTrainer, LatentDiffusion):
    
    pretrained_level = [1, 2, 3, 4, 5]
    level_ckpt_slot = {
        1: "1_2_16",
        2: "1_2_8",
        3: "1_2_4",
        4: "1_2_2",
        5: "1_2_1",
    }
    
    def __init__(
        self, 
        level: int,
        *args,
        sync_path: str = f"{PRETRAINED_CACHE_DIR}/sd_2.1/v2-1_512-ema-pruned.ckpt", 
        sync_control: bool = True,
        # test stage
        sampler: str = "ddpm",
        sampling_steps: int = 50,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # latent feature-guided compression module
        self.preprocess_model = LFGCM(
            in_nc=3,
            enc_mid=[64, 128, 192, 192],
            out_nc=4,
            N=192,
            M=320,
            prior_nc=64,
            sft_ks=3,
            slice_num=10,
            slice_ch=[8,8,8,8,16,16,32,32,96,96]
        )
        
        # conditional diffusion decoding module
        self.control_model = CDDM(
            use_checkpoint=True,
            in_channels=4,
            out_channels=4,
            hint_channels=4,
            model_channels=320,
            attention_resolutions=[4,2,1],
            num_res_blocks=2,
            channel_mult=[1,2,4,4],
            num_head_channels=16,
            use_spatial_transformer=True,
            use_linear_in_transformer=True,
            transformer_depth=1,
            context_dim=1024,
            legacy=False,
            control_scale=1.0,
            control_model_ratio=0.2,
        )
        
        assert sync_path is not None, "Please provide the base model checkpoint path to initialize the control model."
        self.sync_control_weights_from_base_checkpoint(sync_path, sync_control=sync_control)
            
        assert level in self.pretrained_level, f"level should be in {self.pretrained_level}, but got {level}."
        self.level = level
        ckpt_path_pre = f"{PRETRAINED_CACHE_DIR}/DiffEIC/{self.level_ckpt_slot[level]}/lc.ckpt"
        self.load_preprocess_ckpt(ckpt_path_pre=ckpt_path_pre)
        
        self.sampler = sampler
        self.sampling_steps = sampling_steps
        
        self.register_buffer("c_crossattn_buffer", self.get_learned_conditioning([""]).detach(), persistent=False)
    
    def compress(self, imgs: torch.Tensor) -> Dict[str, Any]:
        """Compress the input images into bitstreams.

        Args:
            imgs (torch.Tensor): Target images with shape [B, C, H, W] \
                and value range in [0, 1].

        Returns:
            Dict: containing
                strings (List[List[bytes]]): The compressed bitstream (CompressAI format).
                shape (Tuple[int, int]): The shape of the hyperprior latent.
        """
        z_guidance = self.encode_first_stage(imgs * 2 - 1).mode() * self.scale_factor
        return self.preprocess_model.compress(imgs, z_guidance)
    
    def decompress(self, strings: List[List[bytes]], shape: Tuple[int, int]) -> torch.Tensor:
        """The decompression process of DiffEIC. \
           Decode `z_content` from the bitstream, \
           and then generate images from `z_content` with the latent diffusion model. 

        Args:
            strings (List[List[bytes]]): The compressed bitstream (CompressAI format).
            shape (Tuple[int, int]): The shape of the hyperprior latent.

        Returns:
            torch.Tensor: The generated images with shape [B, C, H, W] and value range in [0, 1].
        """
        z_content = self.preprocess_model.decompress(strings, shape)
        cond = PreprocessOutput(z_content=z_content)
        x_T = torch.randn_like(z_content)
            
        sampler = SpacedSampler(self, var_type="fixed_small") if self.sampler == "ddpm" \
            else DDIMSampler(self)
        
        if isinstance(sampler, SpacedSampler):
            samples = sampler.sample(
                self.sampling_steps, z_content.shape, cond,
                unconditional_guidance_scale=1.0,
                unconditional_conditioning=None,
                cond_fn=None, x_T=x_T
            )
        else:
            sampler: DDIMSampler
            samples, _ = sampler.sample(
                S=self.sampling_steps, batch_size=z_content.shape[0], 
                shape=z_content.shape[1:],
                conditioning=cond, unconditional_conditioning=None,
                x_T=x_T, eta=0
            )
            
        generated_imgs = self.decode_first_stage(samples)
        generated_imgs = ((generated_imgs + 1) / 2).clamp(0, 1)
        
        return generated_imgs

    def apply_model(self, x_noisy, timesteps, cond: PreprocessOutput):
        """Used in sampling process of the sampler. Estimate the one-step noise"""

        N = x_noisy.shape[0]
        
        eps = self.control_model(
            x=x_noisy, timesteps=timesteps,
            context=self.c_crossattn_buffer.repeat_interleave(repeats=N, dim=0),
            hint=cond.z_content,
            base_model=self.model.diffusion_model
        )
        
        return eps
    
    def on_test_start(self):
        self.preprocess_model.update()
        self.freeze()
        
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        imgs, filename, dataset = batch["image"], batch["filename"][0], batch["dataset"][0]
        
        model_name = f"{dataset}/{self.level}"
        
        with self.timer("compress", model_name):
            out_compress = self.compress(imgs)
            if self.ext_params.SaveBitstream:
                self.write_bitstream(filename, **out_compress)
                
        with self.timer("decompress", model_name):
            if self.ext_params.SaveBitstream:
                out_compress = self.read_bitstream(filename)
                    
            preds = self.decompress(**out_compress)
        
        metrics = self.metrics_collector.forward(
            imgs, preds, 
            strings=out_compress["strings"],
            psnr=True, ms_ssim=True, lpips=True, dists=True, kid=True, fid=True
        )

        self.log_test_metrics(
            {
                "bpp": metrics.bpp,
                "psnr": metrics.psnr,
                "ms_ssim": metrics.ms_ssim,
                "lpips": metrics.lpips,
                "dists": metrics.dists,
            },
            {
                "kid": metrics.kid,
                "fid": metrics.fid,
            },
            model_name
        )

        
        if self.ext_params.SaveRecon:
            self.save_recon_imgs(preds, f"{model_name}/{filename}_{metrics.bpp:.4f}.png")
        
    def load_preprocess_ckpt(self, ckpt_path_pre):
        ckpt = torch.load(ckpt_path_pre)
        preprocess_model_ckpt = {}
        control_model_ckpt = {}
        for key in list(ckpt['state_dict'].keys()):
            if "preprocess_model." in key:
                preprocess_model_ckpt[key[17:]] = ckpt['state_dict'][key]
            elif "control_model." in key:
                control_model_ckpt[key[14:]] = ckpt['state_dict'][key]
        
        self.preprocess_model.load_state_dict(preprocess_model_ckpt, strict=True)
        self.control_model.load_state_dict(control_model_ckpt, strict=True)
        logging.info(f'Preprocess model loaded from {ckpt_path_pre}')
        
    def sync_control_weights_from_base_checkpoint(self, path, sync_control=True):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}, please use \
                the `download.sh` in {PRETRAINED_CACHE_DIR}/sd_2.1 to download the base model checkpoints.")
        ckpt_base = torch.load(path, weights_only=False)  # load the base model checkpoints

        if sync_control:
            # add copy for control_module weights from the base model
            for key in list(ckpt_base['state_dict'].keys()):
                if "diffusion_model." in key:
                    if 'control_model.control' + key[15:] in self.state_dict().keys():
                        if ckpt_base['state_dict'][key].shape != self.state_dict()['control_model.control' + key[15:]].shape:
                            if len(ckpt_base['state_dict'][key].shape) == 1:
                                dim = 0
                                control_dim = self.state_dict()['control_model.control' + key[15:]].size(dim)
                                ckpt_base['state_dict']['control_model.control' + key[15:]] = torch.cat([
                                    ckpt_base['state_dict'][key],
                                    ckpt_base['state_dict'][key]
                                ], dim=dim)[:control_dim]
                            else:
                                dim = 0
                                control_dim_0 = self.state_dict()['control_model.control' + key[15:]].size(dim)
                                dim = 1
                                control_dim_1 = self.state_dict()['control_model.control' + key[15:]].size(dim)
                                ckpt_base['state_dict']['control_model.control' + key[15:]] = torch.cat([
                                    ckpt_base['state_dict'][key],
                                    ckpt_base['state_dict'][key]
                                ], dim=dim)[:control_dim_0, :control_dim_1, ...]
                        else:
                            ckpt_base['state_dict']['control_model.control' + key[15:]] = ckpt_base['state_dict'][key]
            
        res_sync = self.load_state_dict(ckpt_base['state_dict'], strict=False)
        logging.warning(f'[{len(res_sync.missing_keys)} keys are missing from the model (hint processing and cross connections included)]')
