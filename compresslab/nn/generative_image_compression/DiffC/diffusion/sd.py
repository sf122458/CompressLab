from abc import ABC, abstractmethod
import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
from .ldm import (
    LatentNoisePredictionModel,
)
from compresslab.utils.constant import PRETRAINED_CACHE_DIR

class SDModel(LatentNoisePredictionModel, ABC):
    def __init__(self, model_id, dtype=torch.float16, cache_dir=PRETRAINED_CACHE_DIR):
        """
        Initialize the SD15 model.
        
        Args:
            device (str): Device to run the model on ("cuda" or "cpu")
            dtype (torch.dtype): Data type for model parameters
        """
        super().__init__()
        
        self.to(dtype)

        # Initialize the pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=self.dtype, cache_dir=cache_dir
        )

        # Store key components for easy access
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer

        # Initialize DDPM scheduler with 1000 steps for SNR calculations
        self.reference_scheduler = DDPMScheduler.from_pretrained(
            model_id, subfolder="scheduler", cache_dir=cache_dir
        )
        self.reference_scheduler.set_timesteps(num_inference_steps=1000)

        # Pre-compute alphas, timesteps, and SNR values for the scheduler
        self.timesteps = self.reference_scheduler.timesteps
        alphas = self.reference_scheduler.alphas_cumprod
        self.register_buffer("snr_values", torch.sqrt(alphas / (1 - alphas)), persistent=False)

        # Initialize configuration attributes
        self.prompt_embeds = None
        self.negative_prompt_embeds = None
        self.guidance_scale = None
        self.image_width = None
        self.image_height = None

    def get_timestep_snr(self, timestep):
        if timestep == 0:
            return torch.inf
        return self.snr_values[timestep - 1]

    def image_to_latent(self, img_pt):
        """
        Convert input image tensor to latent representation.
        
        Args:
            img_pt (torch.Tensor): Image tensor of shape (B, C, H, W) in range [0, 1]
            
        Returns:
            torch.Tensor: Latent representation
        """
        if img_pt.dim() == 3:
            img_pt = img_pt.unsqueeze(0)

        # Move input to correct device and type
        img_pt = img_pt.to(device=self.device, dtype=self.dtype)

        # Encode image to latent space
        latent = self.vae.encode(img_pt * 2 - 1).latent_dist.sample()
        latent = latent * self.vae.config.scaling_factor

        return latent

    def latent_to_image(self, latent):
        """
        Convert latent representation back to image.
        
        Args:
            latent (torch.Tensor): Latent tensor
            
        Returns:
            torch.Tensor: Decoded image in range [0, 1]
        """
        # Scale latents
        latent = latent / self.vae.config.scaling_factor

        # Decode to image space
        with torch.no_grad():
            image = self.vae.decode(latent).sample

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

        # Tokenize and encode the prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)

        # Get prompt embeddings
        with torch.no_grad():
            self.prompt_embeds = self.text_encoder(text_input_ids)[0]

        # For classifier-free guidance, we also need unconditional embeddings
        uncond_tokens = [""] * (
            len([prompt]) if isinstance(prompt, str) else len(prompt)
        )
        uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_input_ids = uncond_input.input_ids.to(self.device)

        with torch.no_grad():
            self.negative_prompt_embeds = self.text_encoder(uncond_input_ids)[0]

    @abstractmethod
    def _get_noise_pred(self, latent_model_input, timestep, encoder_hidden_states):
        """
        Abstract method to get noise prediction from the model.
        
        Args:
            latent_model_input (torch.Tensor): Input latents
            timestep (torch.Tensor): Current timestep
            encoder_hidden_states (torch.Tensor): Text encoder hidden states
            
        Returns:
            torch.Tensor: Predicted noise
        """
        pass

    def predict_noise(self, noisy_latent, timestep):
        """
        Predict noise in the latent at given SNR value.
        
        Args:
            noisy_latent (torch.Tensor): Noisy latent tensor
            snr (float or torch.Tensor): Signal-to-noise ratio
            
        Returns:
            torch.Tensor: Predicted noise
        """

        if self.guidance_scale > 0.0:
            latent_model_input = (
                torch.cat([noisy_latent] * 2).to(self.device).to(self.dtype)
            )

            with torch.no_grad():
                noise_pred = self._get_noise_pred(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=torch.cat(
                        [self.negative_prompt_embeds, self.prompt_embeds]
                    ),
                )

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        else:
            with torch.no_grad():
                noise_pred = self._get_noise_pred(
                    noisy_latent,
                    timestep,
                    encoder_hidden_states=self.negative_prompt_embeds,
                )

        return noise_pred