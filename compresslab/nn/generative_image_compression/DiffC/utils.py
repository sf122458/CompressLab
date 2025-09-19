import torch
from torch import Tensor
from typing import Tuple

def get_alpha_prod_and_beta_prod(snr: float):
    """Map the SNR value to the corresponding alpha_prod and beta_prod values.

    Args:
        snr (float): A SNR value in SNR schedule.

    Returns:
        Tuple:
            - alpha_prod (float): 
            - beta_prod (float): 
    """
    if snr == torch.inf:
        alpha_prod = 1
    else:
        alpha_prod = snr ** 2 / (1 + snr ** 2)
    beta_prod = 1 - alpha_prod
    return alpha_prod, beta_prod

def P(noisy_latent: Tensor, noise_prediction: Tensor, current_snr: float, prev_snr: float) -> Tuple[Tensor, Tensor]:
    """Predict p(x_{t-1} | x_t) given the noisy latent x_t and the predicted noise.

    Args:
        noisy_latent (Tensor): Noisy latent at current timestep, i.e. x_t.
        noise_prediction (Tensor): Noise predicted by the model.
        current_snr (float): The current SNR value in SNR schedule.
        prev_snr (float): The previous SNR value in SNR schedule.

    Returns:
        Tuple:
            pred_prev_sample (Tensor): predicted mean of previous latent
            std (Tensor): predicted standard deviation of previous latent
    """
    alpha_prod_t, beta_prod_t = get_alpha_prod_and_beta_prod(current_snr)
    alpha_prod_t_prev, beta_prod_t_prev = get_alpha_prod_and_beta_prod(prev_snr)
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    pred_original_sample = (
        noisy_latent - beta_prod_t ** (0.5) * noise_prediction
    ) / alpha_prod_t ** (0.5)

    pred_original_sample_coeff = (
        alpha_prod_t_prev ** (0.5) * current_beta_t
    ) / beta_prod_t
    current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

    # 5. Compute predicted previous sample µ_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_prev_sample = (
        pred_original_sample_coeff * pred_original_sample
        + current_sample_coeff * noisy_latent
    )

    # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
    variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
    std = variance ** (0.5)
    return pred_prev_sample, std

def Q(noisy_latent: Tensor, target_latent: Tensor, current_snr: float, prev_snr: float) -> Tensor:
    """q(x_{t-1} | x_t, x_0), i.e. the posterior distribution of x_{t-1} given x_t and x_0.

    Args:
        noisy_latent (Tensor): _description_
        target_latent (Tensor): _description_
        current_snr (float): The current SNR value in SNR schedule.
        prev_snr (float): The previous SNR value in SNR schedule.

    Returns:
        _type_: _description_
    """
    alpha_prod_t, beta_prod_t = get_alpha_prod_and_beta_prod(current_snr)
    alpha_prod_t_prev, beta_prod_t_prev = get_alpha_prod_and_beta_prod(prev_snr)
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    pred_original_sample_coeff = (
        alpha_prod_t_prev ** (0.5) * current_beta_t
    ) / beta_prod_t
    current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

    mu = (
        pred_original_sample_coeff * target_latent + current_sample_coeff * noisy_latent
    )

    return mu

