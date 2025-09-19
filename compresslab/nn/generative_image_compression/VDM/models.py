"""
Modified based on https://github.com/addtt/variational-diffusion-models
"""
from compresslab.nn.base import BasicTrainer
import numpy as np
import torch, math, os
from torch import argmax, autograd, exp, linspace, sigmoid, sqrt, Tensor
from torch.special import expm1
from rich.progress import Progress
from typing import Tuple
from torchvision.utils import save_image
from compresslab.nn.generative_image_compression.VDM.utils import unsqueeze_right, kl_std_normal
from compresslab.nn.generative_image_compression.VDM.unet import UNetVDM
from compresslab.nn.generative_image_compression.VDM.scheduler import FixedLinearSchedule, LearnedLinearSchedule

class VDM(BasicTrainer):
    """Pipeline for variational diffusion models.
    """
    def __init__(
        self, 
        image_shape: Tuple, # TODO

        # vdm unet config
        embedding_dim: int = 128,
        n_blocks: int = 32,
        n_attention_heads: int = 1,
        dropout_prob: float = 0.1,
        norm_groups: int = 32,
        input_channels: int = 3,
        use_fourier_features: bool = True,
        attention_everywhere: bool = False,
        
        noise_schedule: str = "fixed_linear",
        gamma_min: float = -13.3,
        gamma_max: float = 5.0,
        antithetic_time_sampling: bool = True,
        clip_grad_norm: bool = True,
        
        # eval config
        num_samples: int = 64,
        sampling_batch: int = 64,
        n_sample_steps: int = 250,
        clip_samples: bool = True,
        
        **kwargs
    ):
        """
        Args:
            image_shape (Tuple): shape contains (C, H, W)
            n_blocks (int, optional): _description_. Defaults to 32.
            n_attention_heads (int, optional): _description_. Defaults to 1.
            dropout_prob (float, optional): _description_. Defaults to 0.1.
            norm_groups (int, optional): _description_. Defaults to 32.
            input_channels (int, optional): _description_. Defaults to 3.
            use_fourier_features (bool, optional): _description_. Defaults to True.
            attention_everywhere (bool, optional): _description_. Defaults to False.
            noise_schedule (str, optional): _description_. Defaults to "fixed_linear".
            gamma_min (float, optional): _description_. Defaults to -13.3.
            gamma_max (float, optional): _description_. Defaults to 5.0.
            antithetic_time_sampling (bool, optional): _description_. Defaults to True.
            clip_grad_norm (bool, optional): _description_. Defaults to True.
            num_samples (int, optional): _description_. Defaults to 64.
            sampling_batch (int, optional): _description_. Defaults to 64.
            n_sample_steps (int, optional): _description_. Defaults to 250.
            clip_samples (bool, optional): _description_. Defaults to True.
        """
        
        super().__init__(**kwargs)
        
        self.image_shape = image_shape
        self.antithetic_time_sampling = antithetic_time_sampling
        self.clip_grad_norm = clip_grad_norm
        self.num_samples = num_samples
        self.sampling_batch = sampling_batch
        self.n_sample_steps = n_sample_steps
        self.clip_samples = clip_samples
        self.vocab_size = 256  # Number of discrete values per pixel channel
        
        self.diffusion_model = UNetVDM(
            embedding_dim=embedding_dim,
            n_blocks=n_blocks,
            n_attention_heads=n_attention_heads,
            dropout_prob=dropout_prob,
            norm_groups=norm_groups,
            input_channels=input_channels,
            use_fourier_features=use_fourier_features,
            attention_everywhere=attention_everywhere,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )

        if noise_schedule == "fixed_linear":
            self.gamma = FixedLinearSchedule(gamma_min, gamma_max)
        elif noise_schedule == "learned_linear":
            self.gamma = LearnedLinearSchedule(gamma_min, gamma_max)
        else:
            raise ValueError(f"Unknown noise schedule: {noise_schedule}")

    def sample_q_t_0(self, x: Tensor, times: Tensor, noise: Tensor=None):
        """Samples from the distributions q(x_t | x_0) at the given time steps."""
        with torch.enable_grad():  # Need gradient to compute loss even when evaluating
            gamma_t = self.gamma(times)
        gamma_t_padded = unsqueeze_right(gamma_t, x.ndim - gamma_t.ndim)
        mean = x * sqrt(sigmoid(-gamma_t_padded))  # x * alpha
        scale = sqrt(sigmoid(gamma_t_padded))
        if noise is None:
            noise = torch.randn_like(x)
        return mean + noise * scale, gamma_t

    def sample_times(self, batch_size: int) -> Tensor:
        """Note that timestep here is in [0, 1].
        """
        if self.antithetic_time_sampling:
            t0 = np.random.uniform(0, 1 / batch_size)
            times = torch.arange(t0, 1.0, 1.0 / batch_size, device=self.device)
        else:
            times = torch.rand(batch_size, device=self.device)
        return times
    
    def forward(self, x, noise=None):
        bpd_factor = 1 / (np.prod(x.shape[1:]) * np.log(2))

        # Convert image to integers in range [0, vocab_size - 1].
        img_int = torch.round(x * (self.vocab_size - 1)).long()

        # Rescale integer image to [-1 + 1/vocab_size, 1 - 1/vocab_size]
        x = 2 * ((img_int + 0.5) / self.vocab_size) - 1

        # Sample from q(x_t | x_0) with random t.
        times = self.sample_times(x.shape[0]).requires_grad_(True)
        if noise is None:
            noise = torch.randn_like(x)
        x_t, gamma_t = self.sample_q_t_0(x=x, times=times, noise=noise)

        # Forward through model
        model_out = self.diffusion_model(x_t, gamma_t)

        # *** Diffusion loss (bpd)
        gamma_grad = autograd.grad(  # gamma_grad shape: (B, )
            gamma_t,  # (B, )
            times,  # (B, )
            grad_outputs=torch.ones_like(gamma_t),
            create_graph=True,
            retain_graph=True,
        )[0]
        pred_loss = ((model_out - noise) ** 2).sum((1, 2, 3))  # (B, )
        diffusion_loss = 0.5 * pred_loss * gamma_grad * bpd_factor

        # *** Latent loss (bpd): KL divergence from N(0, 1) to q(z_1 | x)
        gamma_1 = self.gamma(torch.tensor([1.0], device=self.device))
        sigma_1_sq = sigmoid(gamma_1)
        mean_sq = (1 - sigma_1_sq) * x**2  # (alpha_1 * x)**2
        latent_loss = kl_std_normal(mean_sq, sigma_1_sq).sum((1, 2, 3)) * bpd_factor

        # *** Reconstruction loss (bpd): - E_{q(z_0 | x)} [log p(x | z_0)].
        # Compute log p(x | z_0) for all possible values of each pixel in x.
        log_probs = self.log_probs_x_z0(x)  # (B, C, H, W, vocab_size)
        # One-hot representation of original image. Shape: (B, C, H, W, vocab_size).
        x_one_hot = torch.zeros((*x.shape, self.vocab_size), device=self.device)
        x_one_hot.scatter_(4, img_int.unsqueeze(-1), 1)  # one-hot over last dim
        # Select the correct log probabilities.
        log_probs = (x_one_hot * log_probs).sum(-1)  # (B, C, H, W)
        # Overall logprob for each image in batch.
        recons_loss = -log_probs.sum((1, 2, 3)) * bpd_factor

        # *** Overall loss in bpd. Shape (B, ).
        loss = diffusion_loss + latent_loss + recons_loss

        with torch.no_grad():
            gamma_0 = self.gamma(torch.tensor([0.0], device=self.device))
        metrics = {
            "bpd": loss.mean(),
            "diff_loss": diffusion_loss.mean(),
            "latent_loss": latent_loss.mean(),
            "loss_recon": recons_loss.mean(),
            "gamma_0": gamma_0.item(),
            "gamma_1": gamma_1.item(),
        }
        return loss.mean(), metrics

    def log_probs_x_z0(self, x=None, z_0=None):
        """Computes log p(x | z_0) for all possible values of x.

        Compute p(x_i | z_0i), with i = pixel index, for all possible values of x_i in
        the vocabulary. We approximate this with q(z_0i | x_i). Unnormalized logits are:
            -1/2 SNR_0 (z_0 / alpha_0 - k)^2
        where k takes all possible x_i values. Logits are then normalized to logprobs.

        The method returns a tensor of shape (B, C, H, W, vocab_size) containing, for
        each pixel, the log probabilities for all `vocab_size` possible values of that
        pixel. The output sums to 1 over the last dimension.

        The method accepts either `x` or `z_0` as input. If `z_0` is given, it is used
        directly. If `x` is given, a sample z_0 is drawn from q(z_0 | x). It's more
        efficient to pass `x` directly, if available.

        Args:
            x: Input image, shape (B, C, H, W).
            z_0: z_0 to be decoded, shape (B, C, H, W).

        Returns:
            log_probs: Log probabilities of shape (B, C, H, W, vocab_size).
        """
        gamma_0 = self.gamma(torch.tensor([0.0], device=self.device))
        if x is None and z_0 is not None:
            z_0_rescaled = z_0 / sqrt(sigmoid(-gamma_0))  # z_0 / alpha_0
        elif z_0 is None and x is not None:
            # Equal to z_0/alpha_0 with z_0 sampled from q(z_0 | x)
            z_0_rescaled = x + exp(0.5 * gamma_0) * torch.randn_like(x)  # (B, C, H, W)
        else:
            raise ValueError("Must provide either x or z_0, not both.")
        z_0_rescaled = z_0_rescaled.unsqueeze(-1)  # (B, C, H, W, 1)
        x_lim = 1 - 1 / self.vocab_size
        x_values = linspace(-x_lim, x_lim, self.vocab_size, device=self.device)
        logits = -0.5 * exp(-gamma_0) * (z_0_rescaled - x_values) ** 2  # broadcast x
        log_probs = torch.log_softmax(logits, dim=-1)  # (B, C, H, W, vocab_size)
        return log_probs
    
    @torch.no_grad()
    def sample_p_s_t(self, z: Tensor, t: Tensor, s: Tensor):
        """Samples from p(z_s | z_t, x). Used for standard ancestral sampling."""
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)
        c = -expm1(gamma_s - gamma_t)
        alpha_t = sqrt(sigmoid(-gamma_t))
        alpha_s = sqrt(sigmoid(-gamma_s))
        sigma_t = sqrt(sigmoid(gamma_t))
        sigma_s = sqrt(sigmoid(gamma_s))

        pred_noise = self.diffusion_model(z, gamma_t)
        if self.clip_samples:
            x_start = (z - sigma_t * pred_noise) / alpha_t
            x_start.clamp_(-1.0, 1.0)
            mean = alpha_s * (z * (1 - c) / alpha_t + c * x_start)
        else:
            mean = alpha_s / alpha_t * (z - c * sigma_t * pred_noise)
        scale = sigma_s * sqrt(c)
        return mean + scale * torch.randn_like(z)
    
    @torch.no_grad()
    def sample(self, batch_size):
        if self.local_rank == 0:
            z = torch.randn((batch_size, *self.image_shape), device=self.device)
            steps = linspace(1.0, 0.0, self.n_sample_steps + 1, device=self.device)
            
            task = self.progress.add_task("Sampling", total=self.n_sample_steps)
            for i in range(self.n_sample_steps):
                z = self.sample_p_s_t(z, steps[i], steps[i + 1])
                self.progress.update(task, advance=1)
                self.progress.refresh()
            self.progress.update(task, visible=False)
            
            logprobs = self.log_probs_x_z0(z_0=z)  # (B, C, H, W, vocab_size)
            x = argmax(logprobs, dim=-1)  # (B, C, H, W)
            return x.float() / (self.vocab_size - 1)  # normalize to [0, 1]

    @torch.no_grad()
    def sample_images(self, is_ema, test=False):
        samples = []
        for i in range(0, self.num_samples, self.sampling_batch):
            corrected_batch_size = min(self.sampling_batch, self.num_samples - i)
            samples.append(self.sample(corrected_batch_size))
        samples = torch.cat(samples, dim=0)
        
        if not test:
            img_path = os.path.join(
                self.trainer.default_root_dir, 
                "samples",
                f"sample-{'ema-' if is_ema else ''}{self.global_step}.png"
            )
        else:
            img_path = os.path.join(
                self.trainer.default_root_dir, 
                "test_samples",
                f"sample.png"
            )
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        save_image(samples, str(img_path), nrow=int(math.sqrt(self.num_samples)))
        
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        data = batch
        loss, metrics = self(data)
        self.manual_backward(loss)
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.diffusion_model.parameters(), max_norm=1.0
            )
            
        self.bar_metrics(metrics)
        self.log_train_metrics(metrics)
        
        optimizer.step()
    
    def on_validation_start(self):
        self.progress = self.trainer.progress_bar_callback.progress
        
        self.sample_images(is_ema=False)
        
    def validation_step(self, batch, batch_idx):
        return
        data = batch
        self.sample_images(self.ema.ema_model, is_ema=True)
        self.sample_images(is_ema=False)
        _, metrics = self(data)
        self.log_val_metrics(metrics)
        
    def on_test_start(self):
        self.progress: Progress = self.trainer.progress_bar_callback.progress

        self.sample_images(is_ema=False, test=True)
        
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return
        data = batch
        _, metrics = self(data)
        self.log_test_metrics(metrics)
        
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.diffusion_model.parameters(),
            self.ext_params.Lr,
            betas=(0.9, 0.99),
            weight_decay=0.01,
            eps=1e-8,
        )
        