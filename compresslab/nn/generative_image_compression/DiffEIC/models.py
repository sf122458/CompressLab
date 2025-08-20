"""
Paper: Towards Extreme Image Compression with Latent Feature Guidance and Diffusion Prior
https://arxiv.org/pdf/2404.18820
"""

import torch

from typing import Mapping, Any, List, Tuple
import logging
import os
import math
import pyiqa
import einops
import numpy as np
from pathlib import Path
from torchvision.utils import save_image
from compresslab.nn.base.utils import (
    write_body, read_body, filesize
)
from compresslab.nn.generative_image_compression.DiffEIC.compression.lfgcm import LFGCM
from compresslab.nn.generative_image_compression.DiffEIC.reconstruction.cddm import CDDM
from compresslab.nn.generative_image_compression.DiffEIC.reconstruction.sampler import SpacedSampler, DDIMSampler
from compresslab.nn.generative_image_compression.DiffEIC.reconstruction.models import LatentDiffusion
from compresslab.nn.generative_image_compression.DiffEIC.reconstruction.utils import log_txt_as_img, default
from compresslab.nn.base import BasicTrainer
from compresslab.utils.constant import PRETRAINED_CACHE_DIR
from compresslab.utils.config import GeneralCodecExtParams

class DiffEIC(BasicTrainer, LatentDiffusion):
    def __init__(
        self, 
        *args, 
        channels: int = 4,
        control_key: str = "hint",
        learning_rate: float = 1e-4,
        aux_learning_rate: float = 1e-3,
        l_bpp_weight: float = 1.0,
        l_guide_weight: float = 2.0,
        sync_path: str = f"{PRETRAINED_CACHE_DIR}/sd_2.1/v2-1_512-ema-pruned.ckpt", 
        synch_control: bool = True,
        pretrained_target_rate: float = None, # (0.02, 0.04, 0.06, 0.09, 0.12)
        calculate_metrics: Mapping[str, Any] = {
            "psnr": {"type": "psnr", "crop_border": 0, "test_y_channel": False},
            "ms_ssim": {"type": "ms_ssim", "test_y_channel": False},
            "lpips": {"type": "lpips", "better": "lower"}
        },
        # test stage
        sampler: str = "ddpm",
        sampling_steps: int = 50,
        ext_params: GeneralCodecExtParams = GeneralCodecExtParams(),
        **kwargs
    ):
        super().__init__(ext_params=ext_params, *args, **kwargs)
        
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
        
        if sync_path is not None:
            self.sync_control_weights_from_base_checkpoint(sync_path, synch_control=synch_control)
            
        if pretrained_target_rate is not None:
            ckpt_path_pre = f"{PRETRAINED_CACHE_DIR}/DiffEIC/{pretrained_target_rate}/lc.ckpt",
            if ckpt_path_pre is not None:
                self.load_preprocess_ckpt(ckpt_path_pre=ckpt_path_pre)

        self.channels = channels

        self.control_key = control_key

        self.learning_rate = learning_rate
        self.aux_learning_rate = aux_learning_rate
        self.l_bpp_weight = l_bpp_weight
        self.l_guide_weight = l_guide_weight
        
        self.sampler = sampler
        self.sampling_steps = sampling_steps

        self.calculate_metrics = calculate_metrics
        self.metric_funcs = {}
        for _, opt in calculate_metrics.items(): 
            mopt = opt.copy()
            name = mopt.pop('type', None)
            mopt.pop('better', None)
            self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)

    def apply_condition_encoder(self, control, x):
        c_latent, likelihoods, q_likelihoods = self.preprocess_model(control, x)
        return c_latent, likelihoods, q_likelihoods
    
    @torch.no_grad()
    def apply_condition_compress(self, control, stream_path, H, W):
        ref = self.encode_first_stage(control * 2 - 1).mode() * self.scale_factor
        out = self.preprocess_model.compress(control, ref)
        shape = out["shape"]
        with Path(stream_path).open("wb") as f:
            write_body(f, shape, out["strings"])
        size = filesize(stream_path)
        bpp = float(size) * 8 / (H * W)
        return bpp

    @torch.no_grad()
    def apply_condition_decompress(self, stream_path):
        with Path(stream_path).open("rb") as f:
            strings, shape = read_body(f)
        c_latent = self.preprocess_model.decompress(strings, shape)
        return c_latent
    
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, bs=bs, *args, **kwargs) 
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()

        c_latent, likelihoods, q_likelihoods = self.apply_condition_encoder(control, x)
        N , _, H, W = control.shape
        num_pixels = N * H * W
        bpp = sum((torch.log(likelihood).sum() / (-math.log(2) * num_pixels)) for likelihood in likelihoods)
        q_bpp = sum((torch.log(likelihood).sum() / (-math.log(2) * num_pixels)) for likelihood in q_likelihoods)
        return x, dict(c_crossattn=[c], c_latent=[c_latent], bpp=bpp, q_bpp=q_bpp, control=[control])
    
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        cond_hint = torch.cat(cond['c_latent'], 1)

        eps = self.control_model(
            x=x_noisy, timesteps=t, context=cond_txt, hint=cond_hint, base_model=diffusion_model)
        
        return eps
    
    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:  # TODO: drop this option
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, t, *args, **kwargs)
    
    @torch.no_grad()
    def log_images(self, batch, sample_steps=50, bs=2):
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=bs)
        bpp = c["q_bpp"]
        bpp_img = [f'{bpp:2f}']*4
        c_latent = c["c_latent"][0]
        control = c["control"][0]
        c = c["c_crossattn"][0]

        log["hq"] = (self.decode_first_stage(z) + 1) / 2
        log["control"] = control
        log["text"] = (log_txt_as_img((512, 512), bpp_img, size=16) + 1) / 2
        
        samples = self.sample_log(
            cond={"c_crossattn": [c], "c_latent": [c_latent]},
            steps=sample_steps
        )
        x_samples = self.decode_first_stage(samples)
        log["samples"] = (x_samples + 1) / 2

        return log, bpp
    
    @torch.no_grad()
    def sample_log(self, cond, steps):
        sampler = SpacedSampler(self)
        b, c, h, w = cond["c_latent"][0].shape
        shape = (b, self.channels, h, w)

        samples = sampler.sample(
            steps, shape, cond, unconditional_guidance_scale=1.0,
            unconditional_conditioning=None
        )
        return samples
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        params += list(param for name, param in self.preprocess_model.named_parameters() 
                       if not name.endswith('.quantiles'))
        
        opt = torch.optim.AdamW(params, lr=lr)

        aux_lr = self.aux_learning_rate
        aux_params = list(param for name, param in self.preprocess_model.named_parameters() 
                       if name.endswith('.quantiles'))
        aux_opt =  torch.optim.AdamW(aux_params, lr=aux_lr)

        return opt, aux_opt
    
    def p_losses(self, x_start, cond, t, noise=None):
        loss_dict = {}
        prefix = 'T' if self.training else 'V'

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/l_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/l_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_bpp = cond['bpp']
        guide_bpp = cond['q_bpp']
        loss_dict.update({f'{prefix}/l_bpp': loss_bpp.mean()})
        loss_dict.update({f'{prefix}/q_bpp': guide_bpp.mean()})
        loss += self.l_bpp_weight * loss_bpp

        c_latent = cond['c_latent'][0][:,:4,:,:]
        loss_guide = self.get_loss(c_latent, x_start)
        loss_dict.update({f'{prefix}/l_guide': loss_guide.mean()})
        loss += self.l_guide_weight * loss_guide
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    
    def training_step(self, batch, batch_idx):
        opt, aux_opt = self.optimizers()
        opt.zero_grad()
        aux_opt.zero_grad()
        
        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val

        loss, loss_dict = self.shared_step(batch)
        
        self.manual_backward(loss)
        # torch.nn.utils.clip_grad_norm_(self.preprocess_model.parameters(), 1.0)

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
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pass
        # out = []
        # log, bpp = self.log_images(batch, bs=None)
        # out.append(bpp.cpu())
        # # save images
        # save_dir = os.path.join(self.logger.save_dir, "validation", f'{self.global_step}')
        # os.makedirs(save_dir, exist_ok=True)
        # image = log["samples"].detach().cpu()
        # image = image.numpy().squeeze().transpose(1,2,0)
        # image = (image * 255).clip(0, 255).astype(np.uint8)
        # path = os.path.join(save_dir, f'{batch_idx}.png')
        # Image.fromarray(image).save(path)

        # control = log["control"].detach().cpu()
        # control = control.numpy().squeeze().transpose(1,2,0)
        # control = (control * 255).clip(0, 255).astype(np.uint8)

        # metric_data = [img2tensor(image).unsqueeze(0) / 255.0, img2tensor(control).unsqueeze(0) / 255.0]

        # for name, _ in self.calculate_metrics.items():
        #     out.append(self.metric_funcs[name](*metric_data))

        
        # return out

    def on_validation_epoch_end(self, outputs):
        outputs = np.array(outputs)
        avg_out = sum(outputs)/len(outputs)
        self.log("avg_bpp", avg_out[0],
                    prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        for i, (name, _) in enumerate(self.calculate_metrics.items()):
            self.log(f"avg_{name}", avg_out[i+1],
                    prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def on_test_start(self):
        super().on_test_start()
        self.preprocess_model.update()
        self.freeze()
        
        
    
    def test_step(self, batch, batch_idx):
        img, filename = batch["image"], batch["filename"][0]
        stream_dir = os.path.join(
            self.trainer.default_root_dir,
            "bitstreams")
        os.makedirs(stream_dir, exist_ok=True)
        
        preds, bpp = self.process(
            imgs=img,
            steps=self.sampling_steps,
            sampler=self.sampler,
            stream_path=os.path.join(stream_dir, f"{filename}.bin")
        )
        
        # if self.save_recon_imgs:
        output_dir = os.path.join(
            self.trainer.default_root_dir, 
            "recon_imgs",
        )
        os.makedirs(output_dir, exist_ok=True)
        save_image(
            preds,
            os.path.join(
                self.trainer.default_root_dir,
                "recon_imgs",
                f"{filename}_{bpp:.4f}.png"
            )
        )
        
        
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

    @torch.no_grad()
    def process(
        self,
        imgs: torch.Tensor,
        sampler: str,
        steps: int,
        stream_path: str
    ) -> Tuple[List[np.ndarray], float]:
        """
        Apply DiffEIC model on a list of images.
        
        Args:
            imgs (List[np.ndarray]): A list of images (HWC, RGB, range in [0, 255])
            sampler (str): Sampler name.
            steps (int): Sampling steps.
            stream_path (str): Savedir of bitstream
        
        Returns:
            preds (List[np.ndarray]): Restoration results (HWC, RGB, range in [0, 255]).
            bpp
        """
        # n_samples = len(imgs)
        n_samples = imgs.shape[0]
        if sampler == "ddpm":
            sampler = SpacedSampler(self, var_type="fixed_small")
        else:
            sampler = DDIMSampler(self)
        # control = torch.tensor(np.stack(imgs) / 255.0, dtype=torch.float32, device=self.device).clamp_(0, 1)
        # control = einops.rearrange(control, "n h w c -> n c h w").contiguous()
        control = imgs
        
        height, width = control.size(-2), control.size(-1)
        bpp = self.apply_condition_compress(control, stream_path, height, width)
        cond = {
            "c_latent": [self.apply_condition_decompress(stream_path)],
            "c_crossattn": [self.get_learned_conditioning([""] * n_samples)]
        }
        
        shape = (n_samples, 4, height // 8, width // 8)
        x_T = torch.randn(shape, device=self.device, dtype=torch.float32)
        if isinstance(sampler, SpacedSampler):
            samples = sampler.sample(
                steps, shape, cond,
                unconditional_guidance_scale=1.0,
                unconditional_conditioning=None,
                cond_fn=None, x_T=x_T
            )
        else:
            sampler: DDIMSampler
            samples, _ = sampler.sample(
                S=steps, batch_size=shape[0], shape=shape[1:],
                conditioning=cond, unconditional_conditioning=None,
                x_T=x_T, eta=0
            )
        
        x_samples = self.decode_first_stage(samples)
        x_samples = ((x_samples + 1) / 2).clamp(0, 1)
        
        return x_samples, bpp
