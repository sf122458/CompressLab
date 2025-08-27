import torch
from tqdm import tqdm

def get_alpha_prod_and_beta_prod(snr):
    if snr == torch.inf:
        alpha_prod = 1
    else:
        alpha_prod = snr ** 2 / (1 + snr ** 2)
    beta_prod = 1 - alpha_prod
    return alpha_prod, beta_prod

def P(noisy_latent, noise_prediction, current_snr, prev_snr):
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

def Q(noisy_latent, target_latent, current_snr, prev_snr):
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

@torch.no_grad()
def encode(
    target_latent,
    timestep_schedule,
    noise_prediction_model,
    gaussian_channel_simulator,
    manual_dkl_per_step=None,
    recon_timesteps=[],
    seed=0,
):
    """Creates a compressed representation of an image using a diffusion model.

    Args:
        target_latent: Latent representation of the image to encode, as produced by the
            diffusion model's VAE encoder.
        timestep_schedule: List of timesteps, parallel to SNR_schedule. The timesteps should match the SNRs that the diffusion model expects at those timesteps.
        SNR_schedule: List of signal-to-noise ratios, decreasing towards zero (e.g.,
            [0.8, 0.6, 0.4, 0.2, 0.1]). SNR values must be in the set of values expected
            by the predict_noise function. Last element must be > 0. Ending with '0'
            (lossless compression of the latent) is not currently supported (and probably
            not desirable).
        predict_noise: Callable which takes in a noisy latent and that latent's SNR, and
            returns a prediction of the latent's noise component.
        gaussian_channel_simulator: Used for gaussian channel simulation.
        manual_dkl_per_step: Used to manually hard-code the dkl per step. Otherwise we'd
            need to send it as side-information. TODO: fancier entropy models of dkl per
            step?
        recon_timesteps: List of timesteps in decreasing order. When used, saves the noisy
            latents from the encoding process at each timestep.
        seed:
            random seed for the compression process.

    Returns:
        tuple:
            - chunk_seeds_per_step (List[List[int]]): One list of ints per step. This is
              the compressed representation of the image, although it still needs to be
              entropy coded. Fed back into the gaussian channel simulator for decoding.
            - dkl_per_step (List[float]): This is also fed back in to the gaussian
              channel simulator to reconstruct the denoising process.
            - noisy_recons: Noisy reconstructions of the target image generated during
              the encoding process. These will be the same noisy reconstructions
              generated during decoding. For faster evaluation, we can skip decoding and
              just use these recons.
            - noisy_recon_step_indices (List[float]): List which is parallel to
              noisy_recons, and reports the step index for each recon.
    """
    chunk_seeds_per_step = []
    dkl_per_step = []
    noisy_recons = []
    noisy_recon_step_indices = []
    recon_timesteps = recon_timesteps.copy()

    torch.manual_seed(seed)
    noisy_latent = torch.randn(
        target_latent.shape, device=target_latent.device, dtype=target_latent.dtype
    )

    current_timestep = 1000
    current_snr = noise_prediction_model.get_timestep_snr(current_timestep)

    for step_index, prev_timestep in tqdm(
        enumerate(timestep_schedule), total=len(timestep_schedule)
    ):  # "previous" as in closer to 1 than the current snr
        noise_prediction = noise_prediction_model.predict_noise(
            noisy_latent, current_timestep
        )
        prev_snr = noise_prediction_model.get_timestep_snr(prev_timestep)
        p_mu, std = P(noisy_latent, noise_prediction, current_snr, prev_snr)
        q_mu = Q(noisy_latent, target_latent, current_snr, prev_snr)
        q_mu_flat_normed = ((q_mu - p_mu) / std).flatten().detach().cpu().numpy()

        manual_dkl = (
            None if manual_dkl_per_step is None else manual_dkl_per_step[step_index]
        )

        sample, chunk_seeds, dkl = gaussian_channel_simulator.encode(
            q_mu_flat_normed, manual_dkl=manual_dkl, seed=step_index
        )
        chunk_seeds_per_step.append(chunk_seeds)
        dkl_per_step.append(dkl)
        sample = torch.tensor(sample)
        reshaped_sample = (
            sample.reshape(noisy_latent.shape)
            .to(noisy_latent.device)
            .to(noisy_latent.dtype)
        )
        noisy_latent = reshaped_sample * std + p_mu
        current_timestep = prev_timestep
        current_snr = prev_snr

        ## Optionally, save the current reconstruction
        save_current_latent = False
        while len(recon_timesteps) > 0 and current_timestep <= recon_timesteps[0]:
            save_current_latent = True
            recon_timesteps = recon_timesteps[1:]

        if save_current_latent:
            noisy_recons.append(noisy_latent)
            noisy_recon_step_indices.append(step_index)

    return chunk_seeds_per_step, dkl_per_step, noisy_recons, noisy_recon_step_indices



@torch.no_grad()
def decode(
        image_width,
        image_height,
        timestep_schedule,
        noise_prediction_model,
        gaussian_channel_simulator,
        chunk_seeds_per_step,
        Dkl_per_step,
        seed,
    ):
    """Decodes a compressed image representation back into its latent space form.

    Args:
        latent_shape (torch.Size): Shape of the target latent tensor to be reconstructed.
        timestep_schedule (List[float]): List of timesteps in decreasing order.
        predict_noise (callable): Function that predicts the noise component given a noisy
            latent and its SNR.
        gaussian_channel_simulator: Simulator used for gaussian channel reconstruction.
        chunk_seeds_per_step (List[List[int]]): Compressed representation of the image,
            consisting of lists of integer seeds for each denoising step.
        Dkl_per_step (List[float]): List of Kullback-Leibler divergence values per step,
            used to reconstruct the denoising process.
        seed (int): Random seed for reproducibility of the denoising process.

    Returns:
        torch.Tensor: The reconstructed latent representation of the image, obtained
            through progressive denoising steps guided by the compressed representation.
    """


    device = noise_prediction_model.device
    dtype = noise_prediction_model.dtype

    dummy_image = torch.zeros((1, 3, image_height, image_width)).to(device).to(dtype)
    dummy_latent = noise_prediction_model.image_to_latent(dummy_image)
    
    torch.manual_seed(seed)
    noisy_latent = torch.randn(dummy_latent.shape, device=device, dtype=dtype)

    current_timestep = 1000
    current_snr = noise_prediction_model.get_timestep_snr(current_timestep)
    for step_index, (prev_timestep, chunk_seeds, Dkl) in tqdm(enumerate(
        zip(timestep_schedule, chunk_seeds_per_step, Dkl_per_step)
    ), total=len(chunk_seeds_per_step)):
        noise_prediction = noise_prediction_model.predict_noise(
            noisy_latent, current_timestep
        )
        prev_snr = noise_prediction_model.get_timestep_snr(prev_timestep)
        p_mu, std = P(noisy_latent, noise_prediction, current_snr, prev_snr)
        sample = gaussian_channel_simulator.decode(
            chunk_seeds, noisy_latent.numel(), Dkl, seed=step_index
        )
        reshaped_sample = (
            torch.tensor(sample).reshape(noisy_latent.shape).to(device).to(dtype)
        )
        noisy_latent = reshaped_sample * std + p_mu
        current_timestep = prev_timestep
        current_snr = prev_snr

    return noisy_latent


@torch.no_grad()
def denoise(noisy_latent, latent_timestep, timestep_schedule, noise_prediction_model):
    """
    Perform probability-flow-based denoising upon the noisy latent.

    Args:
        noisy_latent: latent to be denoised.
        latent_SNR: signal to noise ratio of the latent to be denoised.
        SNR_schedule (List[float]): List of signal-to-noise ratios in decreasing order,
            matching the schedule used during encoding. Last element should be 0 for fully denoised image.
        predict_noise (callable): Function that predicts the noise component given a noisy
            latent and its SNR.

    """
    latent = noisy_latent
    current_timestep = latent_timestep
    current_snr = noise_prediction_model.get_timestep_snr(current_timestep)

    timestep_schedule = [t for t in timestep_schedule if t < latent_timestep]

    for prev_timestep in tqdm(
        timestep_schedule
    ):  # "previous" as in higher than the current snr
        noise_prediction = noise_prediction_model.predict_noise(
            latent.to(noise_prediction_model.dtype), current_timestep
        ).to(torch.float32)
        prev_snr = noise_prediction_model.get_timestep_snr(prev_timestep)

        alpha_prod_t, beta_prod_t = get_alpha_prod_and_beta_prod(current_snr)
        alpha_prod_t_prev, beta_prod_t_prev = get_alpha_prod_and_beta_prod(prev_snr)

        # if int(prev_timestep) == 0:
        #    from IPython.core.debugger import set_trace
        #    set_trace()

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        sample = latent
        model_output = noise_prediction
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        latent = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        )

        current_timestep = prev_timestep
        current_snr = prev_snr

    return latent.to(noisy_latent.dtype)
