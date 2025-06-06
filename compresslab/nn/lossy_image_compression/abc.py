from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import torch, math
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from typing import List, Tuple

@dataclass
class ImageCodecLikelihoods:
    y: torch.Tensor
    z: torch.Tensor = None

    bpp_y: torch.Tensor = field(init=False)
    bpp_z: torch.Tensor = field(init=False)

    def calc_bpp(self, num_pixels):
        self.bpp_y = torch.log(self.y).sum() / (-math.log(2) * num_pixels)
        if self.z is not None:
            self.bpp_z = torch.log(self.z).sum() / (-math.log(2) * num_pixels)
        else:
            self.bpp_z = 0

        return self.bpp_y + self.bpp_z

@dataclass
class ImageCodecForwardInput:
    x: torch.Tensor

@dataclass
class ImageCodecForwardOutput:
    x: torch.Tensor
    x_hat: torch.Tensor
    likelihoods: ImageCodecLikelihoods

    bpp: torch.Tensor = field(init=False)
    mse_loss: torch.Tensor = field(init=False)
    psnr: torch.Tensor = field(init=False)
    ms_ssim: torch.Tensor = field(init=False)

    def __post_init__(self):
        N, _, H, W = self.x.shape
        num_pixels = N * H * W
        if self.likelihoods is not None:
            self.bpp = self.likelihoods.calc_bpp(num_pixels)

        self.mse_loss = F.mse_loss(self.x_hat, self.x)
        self.psnr = 10 * torch.log10(1.0 / self.mse_loss)
        self.ms_ssim = ms_ssim(self.x_hat, self.x, data_range=1.0, size_average=True)


@dataclass
class ImageCodecCompressInput:
    """
    Attributes:
        x (torch.Tensor): Input image tensor to compress.
    """
    x: torch.Tensor


@dataclass
class ImageCodecCompressOutput:
    x: torch.Tensor
    y_strings: List[bytes]
    z_strings: List[bytes] = field(default_factory=list)
    shape: Tuple[int, int] = field(default_factory=lambda: (0, 0))

    bpp: float = field(init=False)

    def __post_init__(self):
        N, _, H, W = self.x.shape
        num_pixels = N * H * W
        total_bits = sum(len(s) * 8 for s in self.y_strings + self.z_strings)
        self.bpp = total_bits / num_pixels

@dataclass
class ImageCodecDecompressOutput:
    x_hat: torch.Tensor

class ImageCodec(ABC):
    @abstractmethod
    def forward(self, input: ImageCodecForwardInput) -> ImageCodecForwardOutput:
        """
        Forward pass for image encoding/decoding.
        
        Args:
            x (torch.Tensor): Input image tensor.
        
        Returns:
            ImageCodecForwardOutput: Output containing the encoded or decoded image.
        """
        pass

    @abstractmethod
    def compress(self, input: ImageCodecCompressInput) -> ImageCodecCompressOutput:
        """
        Compress the input image.
        
        Args:
            input (ImageCodecCompressInput): Input containing the image to compress.
        
        Returns:
            ImageCodecCompressOutput: Output containing compressed data and metadata.
        """
        pass


    @abstractmethod
    def decompress(self, input: ImageCodecCompressOutput) -> ImageCodecDecompressOutput:
        """
        Decompress the input data.
        
        Args:
            input (ImageCodecCompressOutput): Input containing compressed data.
        
        Returns:
            ImageCodecDecompressOutput: Output containing the decompressed image.
        """
        pass