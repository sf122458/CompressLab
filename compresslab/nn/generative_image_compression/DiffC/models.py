"""
DiffC
Modified based on https://github.com/JeremyIV/diffc
Paper: https://arxiv.org/pdf/2501.09815
"""
import zlib, struct, os, logging, sys
from typing import List
from compresslab.nn.base import BasicTrainer
from compresslab.nn.generative_image_compression.DiffC.diffusion import *
from compresslab.nn.generative_image_compression.DiffC.blip import BlipCaptioner
from compresslab.nn.generative_image_compression.DiffC.utils import *
from compresslab.nn.generative_image_compression.DiffC.rcc.gaussian_channel_simulator import GaussianChannelSimulator

class DiffC(BasicTrainer):
    
    supported_models = ["SD1.5", "SD2.1", "SDXL", "Flux"]
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
    
    recon_timesteps_preset = [900, 800, 700, 600, 500, 400, 300, 200, 100, 
                       90, 80, 70, 60, 50, 40, 30, 20, 10]
    
    denoising_timesteps_preset = [981, 961, 941, 921, 901, 881, 861, 841, 821, 801, 
                           781, 761, 741, 721, 701, 681, 661, 641, 621, 601, 
                           581, 561, 541, 521, 501, 481, 461, 441, 421, 401, 
                           381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 
                           181, 161, 141, 121, 101, 81, 61, 41, 21, 10, 5, 0]
    
    def __init__(self, 
                 model_name: str,
                 max_chunk_size: int = 16,
                 chunk_padding: int = 2,
                 encoding_guidance_scale: float = 0,
                 denoising_guidance_scale: float = 0,
                 manual_dkl_per_step = None,
                 recon_timestep: int = 200,
                 encoding_timesteps: List[int]= None,
                 recon_timesteps: List[int] = None,
                 denoising_timesteps: List[int] = None,
                 **kwargs):
        super().__init__(**kwargs)
        
        if encoding_timesteps is None:
            encoding_timesteps = self.encoding_timesteps_preset
        if recon_timesteps is None:
            recon_timesteps = self.recon_timesteps_preset
        if denoising_timesteps is None:
            denoising_timesteps = self.denoising_timesteps_preset
            
        self.encoding_guidance_scale = encoding_guidance_scale
        self.denoising_guidance_scale = denoising_guidance_scale
        self.encoding_timesteps = encoding_timesteps
        self.manual_dkl_per_step = manual_dkl_per_step
        self.recon_timestep = recon_timestep
        self.recon_timesteps = recon_timesteps
        self.denoising_timesteps = denoising_timesteps
        
        assert model_name in self.supported_models, f"model_name should be in {self.supported_models}, but got {model_name}"
        
        if model_name == "SD1.5":
            self.model = SD15Model()
        elif model_name == "SD2.1":
            self.model = SD21Model()
        elif model_name == "SDXL":
            use_refiner = kwargs.get("use_refiner", False)
            self.model = SDXLModel(use_refiner=use_refiner)
        elif model_name == "Flux":
            self.model = FluxModel()
            
        self.gaussian_channel_simulator = GaussianChannelSimulator(
            max_chunk_size,
            chunk_padding,
        )
        
        if self.encoding_guidance_scale or self.denoising_guidance_scale:
            self.captioner = BlipCaptioner()
        else:
            logging.info("Skipping captioner initialization")
        
    def compress(self, imgs):
        
        imgs = imgs.to(self.model.dtype)
        gt_latent = self.model.image_to_latent(imgs)
        
        caption = ""
        if self.encoding_guidance_scale or self.denoising_guidance_scale:
            caption = self.captioner.generate_caption(imgs)
            # logging.info(f"Generated caption: {caption}")
        
        height, width = imgs.shape[2], imgs.shape[3]
        self.model.configure(
            caption, self.encoding_guidance_scale, width, height
        )
        
        chunk_seeds_per_step, Dkl_per_step, _, recon_step_indices = encode(
            gt_latent,
            self.encoding_timesteps,
            self.model,
            self.gaussian_channel_simulator,
            self.manual_dkl_per_step,
            [self.recon_timestep]
        )
        
        step_idx = recon_step_indices[0]
        bytes_data = self.gaussian_channel_simulator.compress_chunk_seeds(
            chunk_seeds_per_step[: step_idx + 1],
            Dkl_per_step[: step_idx + 1],
        )
        
        return {
            "caption": caption,
            "image_bytes": bytes_data,
            "width": width,
            "height": height,
            "step_idx": step_idx
        }
    
    def decompress(self, caption, image_bytes, width, height, step_idx):
        chunk_seeds_per_step = self.gaussian_channel_simulator.decompress_chunk_seeds(
            image_bytes,
            self.manual_dkl_per_step[:step_idx+1]
        )
        
        timestep = self.encoding_timesteps[step_idx]
        
        self.model.configure(
            caption,
            self.denoising_guidance_scale,
            width,
            height,
        )
        
        noisy_recon = decode(
            width,
            height,
            self.encoding_timesteps,
            self.model,
            self.gaussian_channel_simulator,
            chunk_seeds_per_step,
            self.manual_dkl_per_step,
            seed=0
        )
        
        recon_latent = denoise(
            noisy_recon,
            timestep,
            self.denoising_timesteps,
            self.model
        )
        
        recon_img = self.model.latent_to_image(recon_latent)
        return recon_img
    
    def write_diffc_file(self, filename, caption, image_bytes, width, height, step_idx):
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
            "width": width,
            "height": height,
            "step_idx": step_idx
        }
    
    def test_step(self, batch, batch_idx):
        imgs, filename = batch["image"], batch["filename"][0]
        with self.timer("compress"):
            out_compress = self.compress(imgs)
            
            if self.ext_params.SaveBitstream:
                self.write_diffc_file(f"{filename}", **out_compress)
                
        with self.timer("decompress"):
            if self.ext_params.SaveBitstream:
                out_compress = self.read_diffc_file(f"{filename}")

            preds = self.decompress(**out_compress)
        
        metrics = self.metrics_collector.forward(
            imgs, preds, 
            bytes=out_compress["image_bytes"],
            psnr=True, ms_ssim=True, lpips=True, dists=True, kid=True, fid=True
        )
        
        caption_bpp = sys.getsizeof(zlib.compress(out_compress["caption"].encode())) * 8 / (imgs.shape[2] * imgs.shape[3])

        self.log_test_metrics(
            {
                "bpp": metrics.bpp + caption_bpp,
                "image_bpp": metrics.bpp,
                "caption_bpp": caption_bpp,
                "psnr": metrics.psnr,
                "ms-ssim": metrics.ms_ssim,
                "lpips": metrics.lpips,
                "dists": metrics.dists,
            },
            {
                "kid": metrics.kid,
                "fid": metrics.fid,
            }
        )
        
        if self.ext_params.SaveRecon:
            self.save_recon_imgs(preds, f"{filename}_{metrics.bpp:.4f}.png")
    