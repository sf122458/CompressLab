from .sd import SDModel
import torch


class SD15Model(SDModel):
    def __init__(self, device="cuda", dtype=torch.float16):
        super().__init__(
            model_id="runwayml/stable-diffusion-v1-5", device=device, dtype=dtype
        )

    def _get_noise_pred(self, latent_model_input, timestep, encoder_hidden_states):
        """
        Get noise prediction from SD 1.5 UNet (which directly outputs epsilon/noise).
        """
        return self.unet(
            latent_model_input, timestep, encoder_hidden_states=encoder_hidden_states
        ).sample