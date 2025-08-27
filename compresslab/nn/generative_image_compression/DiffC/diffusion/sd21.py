from .sd import SDModel
import torch


class SD21Model(SDModel):
    def __init__(self, dtype=torch.float16):
        super().__init__(
            model_id="stabilityai/stable-diffusion-2-1", dtype=dtype
        )

    def _get_noise_pred(self, latent_model_input, timestep, encoder_hidden_states):
        """
        Get noise prediction from SD 2.1 UNet (converting v-prediction to epsilon).
        """
        # Get v-prediction from model
        v_prediction = self.unet(
            latent_model_input, timestep, encoder_hidden_states=encoder_hidden_states
        ).sample

        # Get alpha and beta values for current timestep
        alpha_prod_t = self.reference_scheduler.alphas_cumprod[timestep - 1]
        beta_prod_t = 1 - alpha_prod_t

        # Convert v-prediction to epsilon (noise) prediction
        noise_pred = (alpha_prod_t ** 0.5) * v_prediction + (
            beta_prod_t ** 0.5
        ) * latent_model_input

        return noise_pred