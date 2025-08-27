from abc import ABC, abstractmethod
import lightning as L

class LatentNoisePredictionModel(L.LightningModule, ABC):
    @abstractmethod
    def image_to_latent(self, img_pt):
        """Convert image to latent representation."""
        pass

    @abstractmethod
    def latent_to_image(self, latent):
        """Convert latent representation to image."""
        pass

    @abstractmethod
    def configure(self, prompt, prompt_guidance, image_width, image_height):
        """Configure the model with given parameters."""
        pass

    @abstractmethod
    def get_timestep_snr(self, timestep):
        """Return the signal to noise ratio (snr) that the model expects at this timestep."""
        pass

    @abstractmethod
    def predict_noise(self, noisy_latent, timestep):
        """Predict noise for given latent at timestep."""
        pass