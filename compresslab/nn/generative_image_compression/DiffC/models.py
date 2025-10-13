"""
DiffC
Modified based on https://github.com/JeremyIV/diffc
Paper: 
[LOSSY COMPRESSION WITH GAUSSIAN DIFFUSION](https://arxiv.org/pdf/2206.08889)
[LOSSY COMPRESSION WITH PRETRAINED DIFFUSION MODELS](https://arxiv.org/pdf/2501.09815)
"""
import zlib, struct, os, logging, sys
from typing import List, Union
from torch import Tensor
from rich.progress import Progress
from compresslab.nn.base import BasicTrainer
from compresslab.nn.generative_image_compression.DiffC.diffusion import *
from compresslab.nn.generative_image_compression.DiffC.blip import BlipCaptioner
from compresslab.nn.generative_image_compression.DiffC.utils import *
from compresslab.nn.generative_image_compression.DiffC.rcc.gaussian_channel_simulator import GaussianChannelSimulator

class DiffC(BasicTrainer):
    
    # hard coding
    max_sample_step = 1000
    supported_models = ["SD1.5", "SD2.1", "SDXL", "FLUX"]
    encoding_timesteps_preset = [972, 949, 929, 897, 869, 834, 805, 780, 751, 726, 
                          704, 688, 670, 648, 627, 608, 591, 578, 561, 546, 
                          530, 520, 510, 498, 491, 480, 465, 455, 447, 438, 
                          429, 419, 410, 402, 390, 380, 371, 361, 353, 345, 
                          336, 326, 319, 313, 305, 296, 289, 282, 276, 269, 
                          261, 254, 247, 242, 237, 231, 224, 219, 213, 209, 
                          204, 200, 194, 189, 185, 181, 175, 170, 167, 163, 
                          160, 156, 153, 149, 146, 143, 139, 135, 132, 129, 
                          125, 121, 118, 116, 113, 110, 107, 104, 101, 99, 
                          96, 94, 92, 90, 87, 85, 82, 80, 78, 76, 74, 72, 
                          70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50, 48, 
                          47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 
                          35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 
                          23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 
                          11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    
    denoising_timesteps_preset = [981, 961, 941, 921, 901, 881, 861, 841, 821, 801, 
                           781, 761, 741, 721, 701, 681, 661, 641, 621, 601, 
                           581, 561, 541, 521, 501, 481, 461, 441, 421, 401, 
                           381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 
                           181, 161, 141, 121, 101, 81, 61, 41, 21, 10, 5, 0]
    
    def __init__(
        self, 
        model_name: str,
        max_chunk_size: int = 16,
        chunk_padding: int = 2,
        encoding_guidance_scale: float = 0,
        denoising_guidance_scale: float = 0,
        manual_dkl_per_step: List[Union[int, float]]= None,
        recon_timestep: int = 200,
        encoding_timesteps: List[int]= None,
        denoising_timesteps: List[int] = None,
        seed: int = 0,
        **kwargs
    ):
        """
        Args:
            model_name (str): The name of the diffusion model to use. 
                Supported models are "SD1.5", "SD2.1", "SDXL", and "Flux".
            max_chunk_size (int, optional): Used in the gaussian channel simulator. Defaults to 16.
            chunk_padding (int, optional): Used in the gaussian channel simulator. Defaults to 2.
            encoding_guidance_scale (float, optional): The guidance scale used in encoding. Defaults to 0.
            denoising_guidance_scale (float, optional): The guidance scale used in denoising. Defaults to 0.
            manual_dkl_per_step (List[Union[int, float]], optional): Pre-computed KL divergence of each step. 
                If it's set to None, the KL divergence will be encoded. Defaults to None.
            recon_timestep (int, optional): The timestep where the latent is encoded with DiffC. Defaults to 200.
            encoding_timesteps (List[int], optional): Custom encoding timesteps. Defaults to None.
            denoising_timesteps (List[int], optional): Custom denoising timesteps. Defaults to None.
            seed (int, optional): The seed used in random coding. Defaults to 0.
        """
        super().__init__(**kwargs)
        
        if encoding_timesteps is None:
            encoding_timesteps = self.encoding_timesteps_preset
        if denoising_timesteps is None:
            denoising_timesteps = self.denoising_timesteps_preset
            
        self.encoding_guidance_scale = encoding_guidance_scale
        self.denoising_guidance_scale = denoising_guidance_scale
        self.encoding_timesteps = encoding_timesteps
        self.manual_dkl_per_step = manual_dkl_per_step
        self.recon_timestep = recon_timestep
        self.denoising_timesteps = denoising_timesteps
        self.seed = seed
        
        model_name = model_name.upper()
        assert model_name in self.supported_models, f"model_name should be in {self.supported_models}, but got {model_name}"
        self.model_name = model_name
        if model_name == "SD1.5":
            self.model = SD15Model()
        elif model_name == "SD2.1":
            self.model = SD21Model()
        elif model_name == "SDXL":
            use_refiner = kwargs.get("use_refiner", False)
            self.model = SDXLModel(use_refiner=use_refiner)
        elif model_name == "FLUX":
            self.model = FluxModel()
            
        self.gaussian_channel_simulator = GaussianChannelSimulator(
            max_chunk_size,
            chunk_padding,
        )
        
        if self.encoding_guidance_scale or self.denoising_guidance_scale:
            self.captioner = BlipCaptioner()
        else:
            logging.info("Skipping captioner initialization")
    
    @torch.no_grad()
    def encode(
        self, 
        target_latent: Tensor,
    ):
        """Perform DiffC on the latent.

        Args:
            target_latent (Tensor): The latent of the origianl image, which is encoded with VAE encoder.

        Returns:
            Tuple:
                - chunk_seeds_per_step (List[List[int]]): The chunk seeds of each step.
                - dkl_per_step (List[float]): The KL divergence of each step.
                - noisy_latent (Tensor): The noisy latent at the recon_timestep.
                - recon_step_index (int): The index of the recon_timestep in the encoding timesteps.
        """
        chunk_seeds_per_step = []
        dkl_per_step = []
        
        torch.manual_seed(self.seed)
        
        noisy_latent = torch.randn_like(target_latent)
        
        current_timestep = self.max_sample_step
        current_snr = self.model.get_timestep_snr(current_timestep)
        
        timestep_schedule = [t for t in self.encoding_timesteps if t >= self.recon_timestep]

        task = self.progress.add_task("Encoding...", total=len(timestep_schedule))
        for step_index, prev_timestep in enumerate(timestep_schedule):
            noise_prediction = self.model.predict_noise(
                noisy_latent, current_timestep
            )
            prev_snr = self.model.get_timestep_snr(prev_timestep)
            p_mu, std = P(noisy_latent, noise_prediction, current_snr, prev_snr)
            q_mu = Q(noisy_latent, target_latent, current_snr, prev_snr)
            q_mu_flat_normed = ((q_mu - p_mu) / std).flatten().detach().cpu().numpy()

            manual_dkl = (
                None if self.manual_dkl_per_step is None else self.manual_dkl_per_step[step_index]
            )

            sample, chunk_seeds, dkl = self.gaussian_channel_simulator.encode(
                q_mu_flat_normed, manual_dkl=manual_dkl, seed=step_index
            )
            
            chunk_seeds_per_step.append(chunk_seeds)
            dkl_per_step.append(dkl)
            sample = torch.tensor(sample).reshape(noisy_latent.shape)\
                .to(noisy_latent.device)\
                .to(noisy_latent.dtype)
            noisy_latent = sample * std + p_mu

            current_timestep = prev_timestep
            current_snr = prev_snr
            
            self.progress.update(task, advance=1)
            self.progress.refresh()
            
        self.progress.update(task, visible=False)

        return chunk_seeds_per_step, dkl_per_step, noisy_latent, len(timestep_schedule)

    @torch.no_grad()
    def decode(
        self,
        shape: Union[torch.Size, List[int]],
        chunk_seeds_per_step: List[List[int]],
    ):
        image_height, image_width = shape[-2:]
        dummy_image = torch.zeros((1, 3, image_height, image_width)).to(self.device).to(self.model.dtype)
        dummy_latent = self.model.image_to_latent(dummy_image)
        
        torch.manual_seed(self.seed)
        noisy_latent = torch.randn_like(dummy_latent)

        current_timestep = self.max_sample_step
        current_snr = self.model.get_timestep_snr(current_timestep)
        
        timestep_schedule = [t for t in self.encoding_timesteps if t >= self.recon_timestep]
        
        task = self.progress.add_task("Decoding...", total=len(timestep_schedule))
        for step_index, (prev_timestep, chunk_seeds, dkl) in enumerate(
            zip(timestep_schedule, chunk_seeds_per_step, self.manual_dkl_per_step)
        ):
            noise_prediction = self.model.predict_noise(
                noisy_latent, current_timestep
            )
            prev_snr = self.model.get_timestep_snr(prev_timestep)
            p_mu, std = P(noisy_latent, noise_prediction, current_snr, prev_snr)
            sample = self.gaussian_channel_simulator.decode(
                chunk_seeds, noisy_latent.numel(), dkl, seed=step_index
            )
            reshaped_sample = (
                torch.tensor(sample).reshape(noisy_latent.shape).to(self.device).to(self.model.dtype)
            )
            noisy_latent = reshaped_sample * std + p_mu
            current_timestep = prev_timestep
            current_snr = prev_snr
            
            self.progress.update(task, advance=1)
            self.progress.refresh()
        self.progress.update(task, visible=False)

        return noisy_latent

    @torch.no_grad()
    def denoise(
        self,
        noisy_latent: Tensor, 
    ) -> Tensor:
        """
        Perform probability-flow-based denoising upon the noisy latent.

        Args:
            noisy_latent: latent to be denoised.
        """
        latent = noisy_latent
        current_timestep = self.recon_timestep
        current_snr = self.model.get_timestep_snr(current_timestep)

        timestep_schedule = [t for t in self.denoising_timesteps_preset if t < self.recon_timestep]

        task = self.progress.add_task("Denoising...", total=len(timestep_schedule))
        for prev_timestep in timestep_schedule:
            noise_prediction = self.model.predict_noise(
                latent.to(self.model.dtype), current_timestep
            ).to(torch.float32)
            prev_snr = self.model.get_timestep_snr(prev_timestep)

            alpha_prod_t, beta_prod_t = get_alpha_prod_and_beta_prod(current_snr)
            alpha_prod_t_prev, _ = get_alpha_prod_and_beta_prod(prev_snr)

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
            
            self.progress.update(task, advance=1)
            self.progress.refresh()

        self.progress.update(task, visible=False)

        return latent.to(noisy_latent.dtype)

        
    def compress(self, imgs):
        # get the latent representation of the input image using the VAE encoder
        imgs = imgs.to(self.model.dtype)
        gt_latent = self.model.image_to_latent(imgs)
        
        # generate caption if guidance scale > 0
        caption = ""
        if self.encoding_guidance_scale or self.denoising_guidance_scale:
            caption = self.captioner.generate_caption(imgs)
        
        height, width = imgs.shape[2], imgs.shape[3]
        self.model.configure(
            caption, self.encoding_guidance_scale, width, height
        )
        
        chunk_seeds_per_step, dkl_per_step, _, recon_step_index = self.encode(
            gt_latent,
        )
        
        step_idx = recon_step_index
        bytes_data = self.gaussian_channel_simulator.compress_chunk_seeds(
            chunk_seeds_per_step[: step_idx + 1],
            dkl_per_step[: step_idx + 1],
        )
        
        return {
            "caption": caption,
            "image_bytes": bytes_data,
            "shape": (height, width),
        }
    
    def decompress(self, image_bytes, shape, caption):
        height, width = shape
        
        chunk_seeds_per_step = self.gaussian_channel_simulator.decompress_chunk_seeds(
            image_bytes,
            self.manual_dkl_per_step
        )
        
        self.model.configure(
            caption,
            self.denoising_guidance_scale,
            width,
            height,
        )
        
        noisy_recon = self.decode(
            shape,
            chunk_seeds_per_step,
        )
        
        
        # denoise the noisy latent and obtain the reconstruction from the latent using the VAE decoder
        recon_latent = self.denoise(
            noisy_recon,
        )
        recon_img = self.model.latent_to_image(recon_latent)
        return recon_img
    
    def write_diffc_file(
        self, 
        filename: str, 
        caption, 
        image_bytes, 
        shape: Union[torch.Size, List[int]],
        step_idx: int
    ):
        height, width = shape
        if not filename.endswith(".bin"):
            filename += ".bin"
            
        bin_path = os.path.join(
            self.trainer.default_root_dir,
            "bitstreams", filename
        )
        os.makedirs(os.path.dirname(bin_path), exist_ok=True)

        # Compress caption with zlib
        compressed_caption = zlib.compress(caption.encode('utf-8'))
        caption_length = len(compressed_caption)

        # Write caption length (4 bytes), width (2 bytes), height (2 bytes), step_idx (2 bytes), 
        # compressed caption, then image data
        with open(bin_path, 'wb') as f:
            f.write(struct.pack('<I', caption_length))  # Write length as 4-byte little-endian uint
            f.write(struct.pack('<H', width))          # Write width as 2-byte little-endian uint
            f.write(struct.pack('<H', height))         # Write height as 2-byte little-endian uint
            f.write(struct.pack('<H', step_idx))       # Write step_idx as 2-byte little-endian uint
            f.write(compressed_caption)
            f.write(bytes(image_bytes))
    
    def read_diffc_file(self, filename: str):
        if not filename.endswith(".bin"):
            filename += ".bin"
        bin_path = os.path.join(
            self.trainer.default_root_dir,
            "bitstreams", filename
        )
        with open(bin_path, 'rb') as f:
            # Read caption length (4 bytes)
            caption_length = struct.unpack('<I', f.read(4))[0]
            
            # Read width, height, and step_idx (2 bytes each)
            width = struct.unpack('<H', f.read(2))[0]
            height = struct.unpack('<H', f.read(2))[0]
            step_idx = struct.unpack('<H', f.read(2))[0]
            
            # Read and decompress caption
            compressed_caption = f.read(caption_length)
            caption = zlib.decompress(compressed_caption).decode('utf-8')
            
            # Read remaining bytes for image data
            image_bytes = list(f.read())
        
        return {
            "caption": caption,
            "image_bytes": image_bytes,
            "shape": (height, width),
            "step_idx": step_idx
        }
        
    def on_test_start(self):
        self.progress: Progress = self.trainer.progress_bar_callback.progress
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        imgs, filename, dataset = batch["image"], batch["filename"][0], batch["dataset"][0]
        model_name = f"{dataset}/DiffC_{self.model_name}_{self.recon_timestep}" #TODO
        with self.timer("compress", model_name):
            out_compress = self.compress(imgs)
            
            if self.ext_params.SaveBitstream:
                self.write_diffc_file(f"{filename}", **out_compress)
                
        with self.timer("decompress", model_name):
            if self.ext_params.SaveBitstream:
                out_compress = self.read_diffc_file(f"{filename}")

            preds = self.decompress(**out_compress)
        
        metrics = self.metrics_collector.forward(
            imgs, preds, 
            strings=bytes(out_compress["image_bytes"]),
            psnr=True, ms_ssim=True, lpips=True, dists=True, kid=True, fid=True
        )
        
        caption_bpp = sys.getsizeof(zlib.compress(out_compress["caption"].encode())) * 8 / (imgs.shape[2] * imgs.shape[3])

        self.log_test_metrics(
            {
                "bpp": metrics.bpp + caption_bpp,
                "image_bpp": metrics.bpp,
                "caption_bpp": caption_bpp,
                "psnr": metrics.psnr,
                "ms_ssim": metrics.ms_ssim,
                "lpips": metrics.lpips,
                "dists": metrics.dists,
            },
            {
                "kid": metrics.kid,
                "fid": metrics.fid,
            },
            model_name
        )
        
        if self.ext_params.SaveRecon:
            self.save_recon_imgs(preds, f"{model_name}/{filename}_{metrics.bpp:.4f}_{metrics.psnr:.2f}_{metrics.lpips:.4f}.png")
    