"""
Paper: Towards Extreme Image Compression with Latent Feature Guidance and Diffusion Prior
https://arxiv.org/pdf/2404.18820
"""

import torch
import torch.nn.functional as F
from typing import Mapping, Any, Dict, Tuple, List
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
    
    ckpt_slot = {
        0.02: "1_2_16",
        0.04: "1_2_8",
        0.06: "1_2_4",
        0.09: "1_2_2",
        0.12: "1_2_1",
    }
    
    def __init__(
        self, 
        *args, 
        channels: int = 4,
        l_simple_weight: float = 1.0,
        l_bpp_weight: float = 1.0,
        l_guide_weight: float = 2.0,
        sync_path: str = f"{PRETRAINED_CACHE_DIR}/sd_2.1/v2-1_512-ema-pruned.ckpt", 
        synch_control: bool = True,
        pretrained_target_rate: float = None, # (0.02, 0.04, 0.06, 0.09, 0.12)
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
            out_nc=channels,
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
        
        if sync_path is not None:
            self.sync_control_weights_from_base_checkpoint(sync_path, synch_control=synch_control)
            
        if pretrained_target_rate is not None:
            self.pretrained_target_rate = pretrained_target_rate
            ckpt_path_pre = f"{PRETRAINED_CACHE_DIR}/DiffEIC/{self.ckpt_slot[pretrained_target_rate]}/lc.ckpt"
            self.load_preprocess_ckpt(ckpt_path_pre=ckpt_path_pre)
        else:
            raise NotImplementedError("Training from scratch is not supported yet.")
        
        self.l_simple_weight = l_simple_weight
        self.l_bpp_weight = l_bpp_weight
        self.l_guide_weight = l_guide_weight
        
        self.sampler = sampler
        self.sampling_steps = sampling_steps
        
        self.register_buffer("c_crossattn_buffer", self.get_learned_conditioning([""]).detach(), persistent=False)
    
    def preprocess_forward(self, imgs: torch.Tensor) -> PreprocessOutput:
        """Provide the image input(range in [0, 1]), obtain `z_content`(the output of the LFGCM),\
            the bpp used to compress `z_content`, and `z_guidance`(the output of the SD VAE encoder).

        Args:
            imgs (torch.Tensor): Target images with shape [B, C, H, W] and value range in [0, 1].

        Returns:
            PreprocessOutput: A dataclass containing:
                - z_guidance (torch.Tensor): The latent representation from the SD VAE encoder.
                - z_content (torch.Tensor): The decoded latent representation from the LFGCM.
                - bpp (torch.Tensor): The bits per pixel of the bitstreams compressed with LFGCM.
                - q_bpp (torch.Tensor): The bits per pixel of the bitstreams compressed with LFGCM using hard quantization.
        """
        N, _, H, W = imgs.shape
        encoder_posterior = self.encode_first_stage(imgs * 2 - 1)
        z_guidance = self.get_first_stage_encoding(encoder_posterior).detach()
        
        z_content, likelihoods, q_likelihoods = self.preprocess_model(imgs, z_guidance)
        
        num_pixels = N * H * W
        bpp = sum((torch.log(likelihood).sum() / (-math.log(2) * num_pixels)) for likelihood in likelihoods)
        q_bpp = sum((torch.log(likelihood).sum() / (-math.log(2) * num_pixels)) for likelihood in q_likelihoods)
        
        return PreprocessOutput(
            z_guidance=z_guidance,
            z_content=z_content,
            bpp=bpp,
            q_bpp=q_bpp,
        )
    
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
    
    @torch.no_grad()
    def generate_imgs(self, cond: PreprocessOutput):
        """Generate images from the output of the LFGCM and the SD VAE encoder.

        Args:
            cond (PreprocessOutput): The conditioning information containing.
        Returns:
            torch.Tensor: The generated images with shape [B, C, H, W] and 
                value range in [0, 1].
        """
        sampler = SpacedSampler(self)
        
        samples = sampler.sample(
            self.sampling_steps, cond.z_content.shape, cond, unconditional_guidance_scale=1.0,
            unconditional_conditioning=None
        )
        
        generated_imgs = self.decode_first_stage(samples)
        generated_imgs = ((generated_imgs + 1) / 2).clamp(0, 1)
        return generated_imgs
    
    def configure_optimizers(self):
        params = list(self.control_model.parameters())
        params += list(param for name, param in self.preprocess_model.named_parameters() 
                       if not name.endswith('.quantiles'))

        opt = torch.optim.AdamW(params, lr=self.ext_params.Lr)

        aux_params = list(param for name, param in self.preprocess_model.named_parameters() 
                       if name.endswith('.quantiles'))
        aux_opt = torch.optim.AdamW(aux_params, lr=self.ext_params.Auxlr)

        return opt, aux_opt
    
    def p_losses(self, cond: PreprocessOutput):
        """Calculate the losses for the diffusion model.

        Args:
            cond (dict): The conditioning information.

        Returns:
            _type_: _description_
        """
        
        loss_dict = {}
        
        z_0 = cond.z_guidance
        t = torch.randint(0, self.num_timesteps, (z_0.shape[0],), device=self.device).long()

        noise = torch.randn_like(z_0)
        
        # diffusion process for the latent representation
        z_t = self.q_sample(x_start=z_0, t=t, noise=noise)
        
        # In eps mode, the UNet predicts the noise. Optimize the z_content here.
        predicted_noise = self.apply_model(z_t, t, cond)
        
        assert self.parameterization == "eps", "Only eps parameterization is supported."

        loss_simple = F.mse_loss(predicted_noise, noise, reduction='none').mean([1, 2, 3])
        loss_dict.update({"l_simple": loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        
        loss_dict.update({"l_gamma": loss.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_bpp = cond.bpp
        guide_bpp = cond.q_bpp
        loss_dict.update({"l_bpp": loss_bpp.mean()})
        loss_dict.update({"q_bpp": guide_bpp.mean()})
        loss += self.l_bpp_weight * loss_bpp

        loss_guide = F.mse_loss(cond.z_content, z_0)
        loss_dict.update({"l_guide": loss_guide.mean()})
        loss += self.l_guide_weight * loss_guide
        loss_dict.update({"loss": loss})

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        # TODO
        raise NotImplementedError("Training is not supported yet.")
        opt, aux_opt = self.optimizers()
        opt.zero_grad()
        aux_opt.zero_grad()
        
        cond = self.preprocess_forward(batch)
        
        loss, loss_dict = self.p_losses(cond)
        
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.preprocess_model.parameters(), 1.0)

        self.log_dict(loss_dict, prog_bar=True,
                    logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                prog_bar=True, logger=True, on_step=True, on_epoch=False)

        
        aux_loss = self.preprocess_model.aux_loss()
        self.log("aux_loss", aux_loss,
                prog_bar=True, logger=True, on_step=True, on_epoch=False)
        
        self.manual_backward(aux_loss)
        opt.step()
        aux_opt.step()
        
    def validation_step(self, batch, batch_idx):
        # TODO
        cond = self.preprocess_forward(batch)
        generated_imgs = self.generate_imgs(cond)
        raise NotImplementedError()
        # self.log(f"generated_imgs/{batch_idx}", generated_imgs)
        

    def on_test_start(self):
        self.preprocess_model.update()
        self.freeze()
        
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        imgs, filename, dataset = batch["image"], batch["filename"][0], batch["dataset"][0]
        
        model_name = f"{dataset}/DiffEIC_{self.pretrained_target_rate}"
        
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
                "ms-ssim": metrics.ms_ssim,
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
        
    def sync_control_weights_from_base_checkpoint(self, path, synch_control=True):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}, please use \
                the `download.sh` in {PRETRAINED_CACHE_DIR}/sd_2.1 to download the base model checkpoints.")
        ckpt_base = torch.load(path, weights_only=False)  # load the base model checkpoints

        if synch_control:
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
