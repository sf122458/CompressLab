import torch
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    AutoencoderKL,
    FluxTransformer2DModel,
)
from accelerate import cpu_offload_with_hook
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from .ldm import LatentNoisePredictionModel
from ..utils import get_alpha_prod_and_beta_prod
import numpy as np
from compresslab.utils.constant import PRETRAINED_CACHE_DIR

def sigma_to_snr(sigma):
    return (1 - sigma) / sigma


def get_ot_flow_to_ddpm_factor(snr):
    OT_flow_noise_sigma = 1 / (snr + 1)

    alpha_cumprod = snr ** 2 / (snr ** 2 + 1)
    DDPM_noise_sigma = torch.sqrt(1 - alpha_cumprod)

    ot_flow_to_ddpm_factor = DDPM_noise_sigma / OT_flow_noise_sigma

    return ot_flow_to_ddpm_factor


class FluxModel(LatentNoisePredictionModel):
    def __init__(
        self,
        model_id="black-forest-labs/FLUX.1-dev",
        dtype=torch.float16,
        cache_dir=PRETRAINED_CACHE_DIR,
    ):
        """
        Initialize the Flux model.
        
        Args:
            model_id (str): HuggingFace model ID
            dtype (torch.dtype): Data type for model parameters
        """
        self.to(dtype)

        # Initialize the pipeline components
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler", cache_dir=cache_dir
        )

        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=self.dtype, cache_dir=cache_dir
        )

        self.transformer = FluxTransformer2DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=self.dtype, cache_dir=cache_dir
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=self.dtype, cache_dir=cache_dir
        )

        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder_2", torch_dtype=self.dtype, cache_dir=cache_dir
        )

        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", cache_dir=cache_dir)

        self.tokenizer_2 = T5TokenizerFast.from_pretrained(
            model_id, subfolder="tokenizer_2", cache_dir=cache_dir
        )

        # Pre-compute SNR values for timesteps
        sigmas = np.arange(1000) / 1000 # Default 1000 timesteps
        snr_values = torch.tensor([sigma_to_snr(sigma) for sigma in sigmas])
        self.register_buffer("snr_values", snr_values, persistent=False)

        # Initialize configuration attributes
        self.prompt_embeds = None
        self.pooled_prompt_embeds = None
        self.text_ids = None
        self.latent_image_ids = None
        self.guidance_scale = None
        self.image_width = None
        self.image_height = None
    
    def enable_model_cpu_offload(self, gpu_id=None, device="cuda"):
        """
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance.
        
        Args:
            gpu_id (int, optional): The ID of the GPU to use. Defaults to None.
            device (str, optional): The device to use. Defaults to "cuda".
        """
        torch_device = torch.device(device)
        if gpu_id is not None and torch_device.index is not None:
            raise ValueError(
                f"You have passed both `gpu_id`={gpu_id} and an index in `device`={device}. "
                "Please specify only one."
            )
        
        # Set the GPU ID to use
        self._offload_gpu_id = gpu_id or torch_device.index or 0
        device = torch.device(f"{torch_device.type}:{self._offload_gpu_id}")
        
        # First move everything to CPU
        self.to("cpu")
        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()
        
        # Set up hooks for the models in sequence
        self._all_hooks = []
        hook = None
        
        # Define sequence of models to offload (matching the Flux pipeline's sequence)
        model_sequence = [
            ("text_encoder", self.text_encoder),
            ("text_encoder_2", self.text_encoder_2),
            ("transformer", self.transformer),
            ("vae", self.vae)
        ]
        
        # Set up CPU offloading hooks for each model in sequence
        for name, model in model_sequence:
            if not isinstance(model, torch.nn.Module):
                continue
                
            # Set up CPU offloading with hook
            _, hook = cpu_offload_with_hook(model, device, prev_module_hook=hook)
            self._all_hooks.append(hook)

    def to(self, device, silence_dtype_warnings=True):
        """
        Moves all models to the specified device.
        
        Args:
            device (str or torch.device): Device to move models to
            silence_dtype_warnings (bool, optional): Whether to silence dtype warnings. Defaults to True.
        """
        self.device = device
        
        # Move all models to device
        if hasattr(self, "text_encoder"):
            self.text_encoder.to(device)
        if hasattr(self, "text_encoder_2"):
            self.text_encoder_2.to(device)
        if hasattr(self, "transformer"):
            self.transformer.to(device)
        if hasattr(self, "vae"):
            self.vae.to(device)
            
        return self

    def get_timestep_snr(self, timestep):
        """Return the SNR value for a given timestep."""
        if timestep == 0:
            return torch.inf
        return self.snr_values[timestep - 1]

    def image_to_latent(self, img_pt):
        """
        Convert input image tensor to latent representation.
        """
        if img_pt.dim() == 3:
            img_pt = img_pt.unsqueeze(0)

        # Move input to correct device and type
        img_pt = img_pt.to(dtype=self.dtype)

        # Encode image to latent space
        vae_latent = self.vae.encode(img_pt * 2 - 1).latent_dist.sample()
        vae_latent = vae_latent * self.vae.config.scaling_factor
        
        # Get dimensions for packing
        batch_size = vae_latent.shape[0]
        num_channels = vae_latent.shape[1]
        height = vae_latent.shape[2]
        width = vae_latent.shape[3]
        
        # Pack the latents into the format expected by transformer
        latent = vae_latent.view(batch_size, num_channels, height // 2, 2, width // 2, 2)        
        latent = latent.permute(0, 2, 4, 1, 3, 5)
        
        latent = latent.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)

        return latent


    def _unpack_latents(self, latents, height, width, vae_scale_factor=8):
        """Unpack transformer format back to VAE latents."""
        # Calculate latent dimensions
        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor
        channels = latents.shape[-1] // 4  # Each patch is 2x2 so divide by 4
        
        # Reshape back to patches
        latents = latents.view(latents.shape[0], latent_height // 2, latent_width // 2, channels, 2, 2)
        # Permute dimensions back to image format
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        # Flatten patch dimensions
        latents = latents.reshape(latents.shape[0], channels, latent_height, latent_width)
        
        return latents

    def latent_to_image(self, latent):
        """Convert packed latent representation back to image."""
        # Unpack latents back to VAE format using helper method
        vae_latent = self._unpack_latents(
            latent,
            height=self.image_height,
            width=self.image_width,
            vae_scale_factor=8
        )
        
        # Scale and shift
        vae_latent = vae_latent / self.vae.config.scaling_factor
        
        # Decode to image space
        with torch.no_grad():
            image = self.vae.decode(vae_latent).sample
            
        return (image / 2 + 0.5).clamp(0, 1).detach()

    def configure(self, prompt, prompt_guidance, image_width, image_height):
        """
        Configure model with prompt and parameters.
        
        Args:
            prompt (str or List[str]): Text prompt(s)
            prompt_guidance (float): Classifier-free guidance scale
            image_width (int): Output image width
            image_height (int): Output image height
        """
        self.guidance_scale = prompt_guidance
        self.image_width = image_width
        self.image_height = image_height

        # Process text with CLIP encoder
        clip_tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            self.pooled_prompt_embeds = self.text_encoder(clip_tokens).pooler_output

        # Process text with T5 encoder
        t5_tokens = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            self.prompt_embeds = self.text_encoder_2(t5_tokens)[0]

        # Prepare text IDs tensor
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        self.text_ids = torch.zeros(batch_size, self.prompt_embeds.shape[1], 3).to(
            device=self.device, dtype=self.dtype
        )

        # Prepare latent image IDs
        # Prepare latent image IDs - exactly matching Flux pipeline's implementation
        height = 2 * (int(self.image_height) // 16)
        width = 2 * (int(self.image_width) // 16)
        
        latent_image_ids = torch.zeros(int(height // 2), int(width // 2), 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, _ = latent_image_ids.shape
        
        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            batch_size, latent_image_id_height * latent_image_id_width, 3
        )

        self.latent_image_ids = latent_image_ids.to(device=self.device, dtype=self.dtype)

    def predict_noise(self, noisy_latent, timestep):
        """
        Predict noise in the latent at given timestep.
        
        Args:
            noisy_latent (torch.Tensor): Noisy latent tensor
            timestep (int): Current timestep
            
        Returns:
            torch.Tensor: Predicted noise in DDPM space
        """
        # Get current SNR for scaling
        snr = self.get_timestep_snr(timestep)

        # Get scaling factor to convert between DDPM and OT flow spaces
        ot_flow_to_ddpm_factor = get_ot_flow_to_ddpm_factor(snr)

        # Convert DDPM space latent to OT flow space
        ot_flow_latent = noisy_latent / ot_flow_to_ddpm_factor

        # Prepare guidance tensor if needed
        guidance = torch.tensor([self.guidance_scale], device=self.device)
        guidance = guidance.expand(noisy_latent.shape[0])

        # Get prediction from transformer (in OT flow space)
        #from IPython.core.debugger import set_trace
        #set_trace()

        with torch.no_grad():
            ot_flow_noise_pred = self.transformer(
                hidden_states=ot_flow_latent,
                timestep=torch.tensor([timestep / 1000], device=self.device),
                guidance=guidance,
                pooled_projections=self.pooled_prompt_embeds,
                encoder_hidden_states=self.prompt_embeds,
                txt_ids=self.text_ids,
                img_ids=self.latent_image_ids,
                return_dict=False,
            )[0].to(torch.float32)

        # TODO: this code is needlessly complicated, because I wanted to avoid doing math.
        # clean it up.
        # TODO: calculate x0 hat in OT flow space:
        sigma = 1 / (snr + 1)
        alpha_prod_t, beta_prod_t = get_alpha_prod_and_beta_prod(snr)
        x0_hat = ot_flow_latent - sigma * ot_flow_noise_pred

        # back-calculate the DDPM noise pred from noisy_latent and x0_hat
        ddpm_noise_pred = (noisy_latent - alpha_prod_t**0.5 * x0_hat) / beta_prod_t**0.5
        # Convert prediction back to DDPM space
        #ddpm_noise_pred = ot_flow_noise_pred * ot_flow_to_ddpm_factor * (alpha_prod_t ** 0.5)

        return ddpm_noise_pred.to(noisy_latent.dtype)