import torch
from torch import nn, Tensor


class Schedule(nn.Module):
    """The log-SNR schedule. Note that the timestep here is normalized to [0, 1].

    The distribution of latent variable $\mathbf{z}_t$ conditioned on the origianl data $\mathbf{x}$, 
    for any $t\in[0,1]$ is given by:
        $$
        q(\mathbf{z}_t|\mathbf{x})=\mathcal{N}(\alpha_t\mathbf{x}, \sigma_t^2\mathbf{I})
        $$
    The signal-to-noise ratio (SNR) is defined as follows:
        $$
        \text{SNR}(t) = \frac{\alpha_t^2}{\sigma_t^2}
        $$
    For variance-preserving diffusion process, we have:
        $$
        \alpha_t = \sqrt{1-\sigma_t^2}
        $$
    For variance-exploding diffusion process, we have:
        $$
        \alpha_t^2=1
        $$
    """
    def __init__(self, clip_samples=True):
        super().__init__()
        self.clip_samples = clip_samples
    
    def gamma(self, timestep):
        raise NotImplementedError
    
    def alpha(self, timestep):
        return torch.sqrt(torch.sigmoid(-self.gamma(timestep)))
    
    def sigma(self, timestep):
        return torch.sqrt(torch.sigmoid(self.gamma(timestep)))
    
    def snr(self, timestep):
        return torch.exp(-self.gamma(timestep))
    
    def denoise_step(self, noisy_image, pred_noise, t, s):
        """The denoising step to obtain $\mathbf{z}_s$ from $\mathbf{z}_t$ using the predicted noise.
            $$
            \hat{\mathbf{x}}_\boldsymbol{\theta}(\mathbf{z}_t;t) = 
            (\mathbf{z}_t - \sigma_t \hat{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}}(\mathbf{z}_t;t)) 
            / \alpha_t
            $$

        Args:
            noisy_image (Tensor): The noisy image at timestep t, i.e., $\mathbf{z}_t$.
            pred_noise (Tensor): Noise predicted by the model, i.e., $\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}}(\mathbf{z}_t;t)$.
            t (Tensor): Timestep for noisy_image.
            s (Tensor): The previous timestep to which we want to denoise.

        Returns:
            Tensor: The denoised image at timestep s, i.e., $\mathbf{z}_s$.
        """
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)
        c = - torch.expm1(gamma_s - gamma_t)
        alpha_t = self.alpha(t)
        alpha_s = self.alpha(s)
        sigma_t = self.sigma(t)
        sigma_s = self.sigma(s)
        
        if self.clip_samples:
            x_start = (noisy_image - sigma_t * pred_noise) / alpha_t
            x_start = x_start.clamp(-1., 1.)
            mean = alpha_s * (noisy_image * (1 - c) / alpha_t + c * x_start)
        else:
            mean = alpha_s / alpha_t * (noisy_image - c * sigma_t * pred_noise)
        scale = sigma_s * torch.sqrt(c)
        return mean + scale * torch.randn_like(noisy_image)
    
    def P(self, noisy_image: Tensor, pred_noise: Tensor, t: Tensor, s: Tensor):
        """This function derives $\mu_{\theta}$ and $\sigma$ from $p(\mathbf{z}_s|\mathbf{z}_t)$. 
        For more details, please refer to Appendix A.3 of the VDM paper.
            
        $\mu$ and $\sigma$ are derived as follows:
            $$
            \mu_{\theta}(\mathbf{z}_t;s,t)=\frac{1}{\alpha_{t|s}}\mathbf{z}_t-
            \frac{\sigma^2_{t|s}}{\alpha_{t|s}\sigma_t}\hat{\boldsymbol{\epsilon}_{\theta}}(\mathbf{z}_t;t)
            $$
            ,
            $$
            \sigma(s,t) = \frac{\sigma_{t|s}\sigma_s}{\sigma_t}
            $$
        where
            $$
            \alpha_{t|s}=\alpha_t/\alpha_s
            $$
        and
            $$
            \sigma_{t|s}^2=\sigma_t^2-\alpha_{t|s}^2\sigma_s^2
            $$
        
        The further simplification of $\mu_{\theta}$ and $\sigma$ can be found in the Appendix A.4:
            $$
            \mu_{\theta}(\mathbf{z}_t;s,t)=\frac{\alpha_s}{\alpha_t}
            (\mathbf{z}_t+\sigma_t\text{expm1}(\gamma_{\boldsymbol{\eta}}(s)-\gamma_{\boldsymbol{\eta}}(t))\hat{
                \boldsymbol{\epsilon}}_{\boldsymbol{\theta}}(\mathbf{z}_t;t))
            $$
            ,
            $$
            \sigma^2(s,t) = \sigma_s ^ 2\cdot (-\text{expm1}(\gamma_\boldsymbol{\eta}(s)-\gamma_\boldsymbol{\eta}(t)))
            $$
            
        
        Args:
            noisy_image (Tensor): The denoised image at timestep t, i.e., $\mathbf{z}_t$.
            pred_noise (Tensor): Noise predicted by the model, i.e., $\hat{\mathbf{\epsilon}}_{\theta}(\mathbf{z}_t;t)$.
            t (Tensor): Timestep for noisy_image.
            s (Tensor): The previous timestep to which we want to denoise.

        Returns:
            Tuple:
                - mu (Tensor): The mean of the Gaussian distribution $p(\mathbf{z}_s|\mathbf{z}_t)$.
                - std (Tensor): The standard deviation of the Gaussian distribution $p(\mathbf{z}_s|\mathbf{z}_t)$.
        """
        alpha_t = self.alpha(t)
        alpha_s = self.alpha(s)
        alpha_t_s = alpha_t / alpha_s
        sigma_t = self.sigma(t)
        sigma_s = self.sigma(s)
        sigma_t_sq = sigma_t ** 2
        sigma_s_sq = sigma_s ** 2
        sigma_t_s_sq = sigma_t_sq - (alpha_t_s ** 2) * sigma_s_sq

        std = sigma_t_s_sq.sqrt() * sigma_s / sigma_t
        mu = noisy_image / alpha_t_s - sigma_t_s_sq * pred_noise / (alpha_t_s * sigma_t)
        
        return mu, std

    def Q(self, noisy_image: Tensor, target_image: Tensor, t: Tensor, s: Tensor):
        """This function derives $\mu$ from $q(\mathbf{z}_s|\mathbf{z}_t,\mathbf{x})$.
        For more details, please refer to Appendix A.2 of the VDM paper.

        $\mu$ is derived as follows:
            $$
            \mu(\mathbf{z}_t,\mathbf{x};s,t)=\frac{\alpha_{t|s}\sigma_s^2}{\sigma_t^2}\mathbf{z}_t+
            \frac{\alpha_s\sigma_{t|s}^2}{\sigma_t^2}\mathbf{x}
            $$
        where
            $$
            \alpha_{t|s}=\alpha_t/\alpha_s
            $$
        and
            $$
            \sigma_{t|s}^2=\sigma_t^2-\alpha_{t|s}^2\sigma_s^2
            $$
        
        Args:
            noisy_image (Tensor): Noisy image at timestep t, i.e., $\mathbf{z}_t$.
            target_image (Tensor): The original image, i.e., $\mathbf{x}$.
            t (Tensor): Timestep for noisy_image.
            s (Tensor): The previous timestep to which we want to denoise.

        Returns:
            mu (Tensor): The mean of the Gaussian distribution $q(\mathbf{z}_s|\mathbf{z}_t,\mathbf{x})$.
        """
        alpha_t = self.alpha(t)
        alpha_s = self.alpha(s)
        alpha_t_s = alpha_t / alpha_s
        sigma_t = self.sigma(t)
        sigma_s = self.sigma(s)
        sigma_t_sq = sigma_t ** 2
        sigma_s_sq = sigma_s ** 2
        sigma_t_s_sq = sigma_t_sq - (alpha_t_s ** 2) * sigma_s_sq

        mu = alpha_t_s * sigma_s_sq * noisy_image / sigma_t_sq \
            + alpha_s * sigma_t_s_sq * target_image / sigma_t_sq

        return mu

class VariancePreservingSchedule(Schedule):
    """The log-SNR schedule. Note that the timestep here is normalized to [0, 1].

    Note that the VDM paper ultilizes the variance-preserving diffusion process, i.e.,
        $$
        \alpha_t = \sqrt{1-\sigma_t^2}
        $$ 
    The noise schedule is used with parameterization, i.e.
        $$
        \sigma_t^2=\text{sigmoid}(\gamma_{\boldsymbol{\eta}}(t))
        $$
    where $\gamma_{\boldsymbol{\eta}}(t)$ is a learnable or fixed function that maps $t\in[0,1]$ to $\mathbb{R}$.
    Other terms can be simplified as follows:
        $$
        \alpha_t^2=\text{sigmoid}(-\gamma_{\boldsymbol{\eta}}(t))
        $$
        $$
        \text{SNR}(t) = \exp({-\gamma_{\boldsymbol{\eta}}(t)})
        $$
    """
    def __init__(self, clip_samples=True):
        super().__init__()
        self.clip_samples = clip_samples

    def alpha(self, timestep):
        return torch.sqrt(torch.sigmoid(-self.gamma(timestep)))
    
    def sigma(self, timestep):
        return torch.sqrt(torch.sigmoid(self.gamma(timestep)))
    
    def snr(self, timestep):
        return torch.exp(-self.gamma(timestep))
    
    def denoise_step(self, noisy_image, pred_noise, t, s):
        """The simplified denoising step to obtain $\mathbf{z}_s$ from $\mathbf{z}_t$ using the predicted noise.
        For more details, please refer to Appendix A.4 of the VDM paper.
        The simplification of $\mu_{\theta}$ and $\sigma$ in $p(\mathbf{z}_s|\mathbf{z}_t)$ are as follows:
            $$
            \mu_{\theta}(\mathbf{z}_t;s,t)=\frac{\alpha_s}{\alpha_t}
            (\mathbf{z}_t-\sigma_tc\hat{
                \boldsymbol{\epsilon}}_{\boldsymbol{\theta}}(\mathbf{z}_t;t))
            $$
        ,
            $$
            \sigma^2(s,t) = \sigma_s ^ 2\cdot c
            $$

        The ancestral sampling from $p(\mathbf{z}_s|\mathbf{z}_t)$ can be performed as follows:
            $$
            \mathbf{z}_s = \frac{\alpha_s}{\alpha_t}(\mathbf{z}_t-\sigma_t c 
            \hat{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}}(\mathbf{z}_t;t)) 
            + \sigma_s \sqrt{c}\boldsymbol{\epsilon}
            $$
        where $c = -\text{expm1}(\gamma_{\boldsymbol{\eta}}(s)-\gamma_{\boldsymbol{\eta}}(t))$

        Args:
            noisy_image (Tensor): The noisy image at timestep t, i.e., $\mathbf{z}_t$.
            pred_noise (Tensor): Noise predicted by the model, i.e., $\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}}(\mathbf{z}_t;t)$.
            t (Tensor): Timestep for noisy_image.
            s (Tensor): The previous timestep to which we want to denoise.

        Returns:
            Tensor: The denoised image at timestep s, i.e., $\mathbf{z}_s$.
        """
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)
        c = - torch.expm1(gamma_s - gamma_t)
        alpha_t = self.alpha(t)
        alpha_s = self.alpha(s)
        sigma_t = self.sigma(t)
        sigma_s = self.sigma(s)
        
        if self.clip_samples:
            x_start = (noisy_image - sigma_t * pred_noise) / alpha_t
            x_start = x_start.clamp(-1., 1.)
            mean = alpha_s * (noisy_image * (1 - c) / alpha_t + c * x_start)
        else:
            mean = alpha_s / alpha_t * (noisy_image - c * sigma_t * pred_noise)
        scale = sigma_s * torch.sqrt(c)
        return mean + scale * torch.randn_like(noisy_image)
    
    def P(self, noisy_image: Tensor, pred_noise: Tensor, t: Tensor, s: Tensor):
        """This function derives $\mu_{\theta}$ and $\sigma$ from $p(\mathbf{z}_s|\mathbf{z}_t)$. 
        The further simplification of $\mu_{\theta}$ and $\sigma$ can be found in the Appendix A.4 of the VDM paper:
            $$
            \mu_{\theta}(\mathbf{z}_t;s,t)=\frac{\alpha_s}{\alpha_t}
            (\mathbf{z}_t-\sigma_tc\hat{
                \boldsymbol{\epsilon}}_{\boldsymbol{\theta}}(\mathbf{z}_t;t))
            $$
        ,
            $$
            \sigma^2(s,t) = \sigma_s ^ 2\cdot c
            $$
        
        Args:
            noisy_image (Tensor): The denoised image at timestep t, i.e., $\mathbf{z}_t$.
            pred_noise (Tensor): Noise predicted by the model, i.e., $\hat{\mathbf{\epsilon}}_{\theta}(\mathbf{z}_t;t)$.
            t (Tensor): Timestep for noisy_image.
            s (Tensor): The previous timestep to which we want to denoise.

        Returns:
            Tuple:
                - mu (Tensor): The mean of the Gaussian distribution $p(\mathbf{z}_s|\mathbf{z}_t)$.
                - std (Tensor): The standard deviation of the Gaussian distribution $p(\mathbf{z}_s|\mathbf{z}_t)$.
        """
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)
        c = - torch.expm1(gamma_s - gamma_t)
        alpha_t = self.alpha(t)
        alpha_s = self.alpha(s)
        sigma_t = self.sigma(t)
        sigma_s = self.sigma(s)
        
        mean = alpha_s / alpha_t * (noisy_image - c * sigma_t * pred_noise)
        std = sigma_s * torch.sqrt(c)
        
        return mean, std

        

class FixedLinearSchedule(VariancePreservingSchedule):
    def __init__(self, gamma_min: float, gamma_max: float, **kwargs):
        super().__init__(**kwargs)
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def gamma(self, t: Tensor) -> Tensor:
        return self.gamma_min + (self.gamma_max - self.gamma_min) * t


class LearnedLinearSchedule(VariancePreservingSchedule):
    def __init__(self, gamma_min: float, gamma_max: float, **kwargs):
        super().__init__(**kwargs)
        self.b = nn.Parameter(torch.tensor(gamma_min))
        self.w = nn.Parameter(torch.tensor(gamma_max - gamma_min))

    def gamma(self, t: Tensor) -> Tensor:
        return self.b + self.w.abs() * t
    
class LearnedNNSchedule(VariancePreservingSchedule):
    def __init__(self, gamma_min: float, gamma_max: float, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def gamma(self, t: Tensor) -> Tensor:
        raise NotImplementedError("This function is not implemented yet.")
        t = t.unsqueeze(-1)  # Add feature dimension
        gamma = self.net(t).squeeze(-1)  # Remove feature dimension
        return self.gamma_min + (self.gamma_max - self.gamma_min) * gamma