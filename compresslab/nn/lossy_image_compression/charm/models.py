"""Pytorch implementation of the ChARM model for lossy image compression.

Official implementation: https://github.com/tensorflow/compression/blob/master/models/ms2020.py#L137
Paper: https://arxiv.org/pdf/2007.08739v1
"""

from compresslab.core.models import CompressionModel
from compresslab.core.entropy_models import EntropyBottleneck, GaussianConditional
from compresslab.core.layers import GDN, conv3x3, conv, deconv
from compresslab.nn.lossy_image_compression.abc import (
    ImageCodec,
    ImageCodecForwardInput,
    ImageCodecForwardOutput,
    ImageCodecCompressInput,
    ImageCodecCompressOutput,
    ImageCodecLikelihoods,
    ImageCodecDecompressOutput)
import torch.nn as nn
import torch

class ChARM(CompressionModel, ImageCodec):
    def __init__(self, N=192, M=320, 
                 num_slices=10, **kwargs):
        super().__init__()

        self.N = int(N)
        self.M = int(M)
        self.num_slices = num_slices
        
        assert M % num_slices == 0, "M must be divisible by num_slices"
        ch_per_slice = M // num_slices
        self.ch_per_slice = ch_per_slice

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M)
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3)
        )

        self.h_a = nn.Sequential(
            conv(M, M, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(M, (N + M) // 2),
            nn.ReLU(inplace=True),
            conv((N + M) // 2, N)
        )

        self.h_s = nn.Sequential(
            deconv(N, (N + M) // 2),
            nn.ReLU(inplace=True),
            deconv((N + M) // 2, M),
            nn.ReLU(inplace=True),
            conv(M, M, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        self.cc_mean_transforms = nn.ModuleList([])
        self.cc_scale_transforms = nn.ModuleList([])
        self.lrp_transforms = nn.ModuleList([])

        for i in range(num_slices):
            self.cc_mean_transforms.append(nn.Sequential(
                conv3x3(M // 2 + i * ch_per_slice, (N + M) // 2),
                nn.ReLU(inplace=True),
                conv3x3((N + M) // 2, N),
                nn.ReLU(inplace=True),
                conv3x3(N, ch_per_slice),
            ))

            self.cc_scale_transforms.append(nn.Sequential(
                conv3x3(M // 2 + i * ch_per_slice, (N + M) // 2),
                nn.ReLU(inplace=True),
                conv3x3((N + M) // 2, N),
                nn.ReLU(inplace=True),
                conv3x3(N, ch_per_slice),
            ))

            self.lrp_transforms.append(nn.Sequential(
                conv3x3(M // 2 + (i + 1) * ch_per_slice, (N + M) // 2),
                nn.ReLU(inplace=True),
                conv3x3((N + M) // 2, N),
                nn.ReLU(inplace=True),
                conv3x3(N, ch_per_slice),
            ))

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, input: ImageCodecForwardInput) -> ImageCodecForwardOutput:
        y = self.g_a(input.x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, dim=1)

        y_slices = y.chunk(self.num_slices, dim=1)
        y_hat_slices = []
        y_likelihoods = []

        for slice_index, y_slice in enumerate(y_slices):
            
            mean = torch.cat([means_hat] + y_hat_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean)

            scale = torch.cat([scales_hat] + y_hat_slices, dim=1)
            sigma = self.cc_scale_transforms[slice_index](scale)

            y_likelihoods.append(
                self.gaussian_conditional._likelihood(y_slice, scales=sigma, means=mu)
            )

            # use straight-through estimator
            y_hat_slice = torch.round(y_slice) + y_slice - y_slice.detach()

            y_hat_slices.append(y_hat_slice)

            lrp_support = torch.cat([mean, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = torch.nn.functional.tanh(lrp) / 2
            y_hat_slice += lrp


        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihoods, dim=1)

        x_hat = self.g_s(y_hat)

        return ImageCodecForwardOutput(
            x=input.x,
            x_hat=x_hat,
            likelihoods=ImageCodecLikelihoods(
                y=y_likelihoods,
                z=z_likelihoods
            )
        )
    
    def compress(self, input: ImageCodecCompressInput) -> ImageCodecCompressOutput:
        pass

    def decompress(self, input: ImageCodecCompressOutput) -> ImageCodecDecompressOutput:
        pass


if __name__ == "__main__":
    model = ChARM()
    x = torch.randn(1, 3, 256, 256)
    out = model(x)