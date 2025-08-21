import torch
import pytorch_lightning as pl
from ..modules.diffusionmodules.model import Encoder, Decoder
from ..modules.distributions.distributions import DiagonalGaussianDistribution
import logging

class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 double_z=True,
                 z_channels=4,
                 resolution=256,
                 in_channels=3,
                 out_ch=3,
                 ch=128,
                 ch_mult=(1, 2, 4, 8),
                 num_res_blocks=2,
                 attn_resolutions=[],
                 dropout=0.0,
                 embed_dim=4,
                 ):
        super().__init__()
        self.ch_mult = ch_mult
        self.encoder = Encoder(z_channels=z_channels,
                               resolution=resolution,
                               in_channels=in_channels,
                               out_ch=out_ch,
                               ch=ch,
                               ch_mult=ch_mult,
                               double_z=double_z,
                               num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions,
                               dropout=dropout)
        self.decoder = Decoder(z_channels=z_channels,
                               resolution=resolution,
                               in_channels=in_channels,
                               out_ch=out_ch,
                               ch=ch,
                               ch_mult=ch_mult,
                               double_z=double_z,
                               num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions,
                               dropout=dropout)
        
        self.quant_conv = torch.nn.Conv2d(2*z_channels, 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)
        self.embed_dim = embed_dim

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    logging.info("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        logging.info(f"Restored from {path}")

    def encode(self, x):
        """Encodes the input image into a posterior distribution.

        Args:
            x (torch.Tensor): The target image with shape [B, C, H, W] 
                and normalized to [-1, 1].

        Returns:
            DiagonalGaussianDistribution: _description_
        """
        h = self.encoder(x)
        # -> [B, 2*z_channels, H, W]
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        """Decodes the latent representation into an image.

        Args:
            z (torch.Tensor): The latent representation 
                with channels size equal to `2*z_channels`.

        Returns:
            torch.Tensor: The generated image.
        """
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior