import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import amp

from compresslab.core.entropy_models import EntropyBottleneck, GaussianConditional
from compresslab.core.layers import QReLU
from compresslab.core.ops import quantize_ste

from compresslab.core.models.base import CompressionModel
from compresslab.core.layers import conv, deconv, gaussian_blur, gaussian_kernel2d, meshgrid2d

from compresslab.nn.video_compression.abc import *


class ScaleSpaceFlow(VideoCodec):
    r"""Google's first end-to-end optimized video compression from E.
    Agustsson, D. Minnen, N. Johnston, J. Balle, S. J. Hwang, G. Toderici: `"Scale-space flow for end-to-end
    optimized video compression" <https://openaccess.thecvf.com/content_CVPR_2020/html/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.html>`_,
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2020).

    Args:
        num_levels (int): Number of Scale-space
        sigma0 (float): standard deviation for gaussian kernel of the first space scale.
    """

    def __init__(
        self,
        num_levels: int = 5,
        sigma0: float = 1.5,
    ):
        super().__init__()

        class Encoder(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, out_planes, kernel_size=5, stride=2),
                )

        class Decoder(nn.Sequential):
            def __init__(
                self, out_planes: int, in_planes: int = 192, mid_planes: int = 128
            ):
                super().__init__(
                    deconv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, out_planes, kernel_size=5, stride=2),
                )

        class HyperEncoder(nn.Sequential):
            def __init__(
                self, in_planes: int = 192, mid_planes: int = 192, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                )

        class HyperDecoder(nn.Sequential):
            def __init__(
                self, in_planes: int = 192, mid_planes: int = 192, out_planes: int = 192
            ):
                super().__init__(
                    deconv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, out_planes, kernel_size=5, stride=2),
                )

        class HyperDecoderWithQReLU(nn.Module):
            def __init__(
                self, in_planes: int = 192, mid_planes: int = 192, out_planes: int = 192
            ):
                super().__init__()

                def qrelu(input, bit_depth=8, beta=100):
                    return QReLU.apply(input, bit_depth, beta)

                self.deconv1 = deconv(in_planes, mid_planes, kernel_size=5, stride=2)
                self.qrelu1 = qrelu
                self.deconv2 = deconv(mid_planes, mid_planes, kernel_size=5, stride=2)
                self.qrelu2 = qrelu
                self.deconv3 = deconv(mid_planes, out_planes, kernel_size=5, stride=2)
                self.qrelu3 = qrelu

            def forward(self, x):
                x = self.qrelu1(self.deconv1(x))
                x = self.qrelu2(self.deconv2(x))
                x = self.qrelu3(self.deconv3(x))

                return x

        class Hyperprior(CompressionModel):
            def __init__(self, planes: int = 192, mid_planes: int = 192):
                super().__init__()
                self.entropy_bottleneck = EntropyBottleneck(mid_planes)
                self.hyper_encoder = HyperEncoder(planes, mid_planes, planes)
                self.hyper_decoder_mean = HyperDecoder(planes, mid_planes, planes)
                self.hyper_decoder_scale = HyperDecoderWithQReLU(
                    planes, mid_planes, planes
                )
                self.gaussian_conditional = GaussianConditional(None)

            def forward(self, y):
                z = self.hyper_encoder(y)
                z_hat, z_likelihoods = self.entropy_bottleneck(z)

                scales = self.hyper_decoder_scale(z_hat)
                means = self.hyper_decoder_mean(z_hat)
                _, y_likelihoods = self.gaussian_conditional(y, scales, means)
                y_hat = quantize_ste(y - means) + means
                return y_hat, {"y": y_likelihoods, "z": z_likelihoods}

            def compress(self, y):
                z = self.hyper_encoder(y)

                z_string = self.entropy_bottleneck.compress(z)
                z_hat = self.entropy_bottleneck.decompress(z_string, z.size()[-2:])

                scales = self.hyper_decoder_scale(z_hat)
                means = self.hyper_decoder_mean(z_hat)

                indexes = self.gaussian_conditional.build_indexes(scales)
                y_string = self.gaussian_conditional.compress(y, indexes, means)
                y_hat = self.gaussian_conditional.quantize(y, "dequantize", means)

                return y_hat, {"strings": [y_string, z_string], "shape": z.size()[-2:]}

            def decompress(self, strings, shape):
                assert isinstance(strings, list) and len(strings) == 2
                z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

                scales = self.hyper_decoder_scale(z_hat)
                means = self.hyper_decoder_mean(z_hat)
                indexes = self.gaussian_conditional.build_indexes(scales)
                y_hat = self.gaussian_conditional.decompress(
                    strings[0], indexes, z_hat.dtype, means
                )

                return y_hat

        self.img_encoder = Encoder(3)
        self.img_decoder = Decoder(3)
        self.img_hyperprior = Hyperprior()

        self.res_encoder = Encoder(3)
        self.res_decoder = Decoder(3, in_planes=384)
        self.res_hyperprior = Hyperprior()

        self.motion_encoder = Encoder(2 * 3)
        self.motion_decoder = Decoder(2 + 1)
        self.motion_hyperprior = Hyperprior()

        self.sigma0 = sigma0
        self.num_levels = num_levels

    def forward_I_frame(self, input: IFrameForwardInput) -> IFrameForwardOutput:
        y = self.img_encoder(input.input_frame)
        y_hat, likelihoods = self.img_hyperprior(y)
        x_hat = self.img_decoder(y_hat)
        return IFrameForwardOutput(
            input_frame=input.input_frame,
            recon_frame=x_hat,
            likelihoods=IFrameLikelihoods(
                y=likelihoods["y"],
                z=likelihoods["z"],
            )
        )

    def compress_I_frame(self, input: IFrameCompressInput) -> IFrameCompressOutput:
        y = self.img_encoder(input.input_frame)
        y_hat, out_keyframe = self.img_hyperprior.compress(y)
        x_hat = self.img_decoder(y_hat)

        return IFrameCompressOutput(
            input_frame=input.input_frame,
            recon_frame=x_hat,
            y_strings=out_keyframe["strings"][0],
            z_strings=out_keyframe["strings"][1],
            shape=out_keyframe["shape"],
        )

    def decompress_I_frame(self, input: IFrameCompressOutput) -> IFrameDecompressOutput:
        y_hat = self.img_hyperprior.decompress(input.strings, input.shape)
        x_hat = self.img_decoder(y_hat)

        return IFrameDecompressOutput(recon_frame=x_hat)

    def forward_P_frame(self, input: PFrameForwardInput) -> PFrameForwardOutput:
        # encode the motion information
        x_cur, x_ref = input.input_frame, input.refer_frame
        x = torch.cat((x_cur, x_ref), dim=1)
        y_motion = self.motion_encoder(x)
        y_motion_hat, motion_likelihoods = self.motion_hyperprior(y_motion)

        # decode the space-scale flow information
        motion_info = self.motion_decoder(y_motion_hat)
        x_pred = self.forward_prediction(x_ref, motion_info)

        # residual
        x_res = x_cur - x_pred
        y_res = self.res_encoder(x_res)
        y_res_hat, res_likelihoods = self.res_hyperprior(y_res)

        # y_combine
        y_combine = torch.cat((y_res_hat, y_motion_hat), dim=1)
        x_res_hat = self.res_decoder(y_combine)

        # final reconstruction: prediction + residual
        x_rec = x_pred + x_res_hat

        # return x_rec, {"motion": motion_likelihoods, "residual": res_likelihoods}
        return PFrameForwardOutput(
            input_frame=x_cur,
            recon_frame=x_rec,
            likelihoods=PFrameLikelihoods(
                y_mv=motion_likelihoods["y"],
                z_mv=motion_likelihoods["z"],
                y=res_likelihoods["y"],
                z=res_likelihoods["z"],
            )
        )

    def compress_P_frame(self, input: PFrameCompressInput) -> PFrameCompressOutput:
        x_cur, x_ref = input.input_frame, input.refer_frame
        # encode the motion information
        x = torch.cat((x_cur, x_ref), dim=1)
        y_motion = self.motion_encoder(x)
        y_motion_hat, out_motion = self.motion_hyperprior.compress(y_motion)

        # decode the space-scale flow information
        motion_info = self.motion_decoder(y_motion_hat)
        x_pred = self.forward_prediction(x_ref, motion_info)

        # residual
        x_res = x_cur - x_pred
        y_res = self.res_encoder(x_res)
        y_res_hat, out_res = self.res_hyperprior.compress(y_res)

        # y_combine
        y_combine = torch.cat((y_res_hat, y_motion_hat), dim=1)
        x_res_hat = self.res_decoder(y_combine)

        # final reconstruction: prediction + residual
        x_rec = x_pred + x_res_hat

        return PFrameCompressOutput(
            input_frame=x_cur,
            recon_frame=x_rec,
            refer_frame=x_ref,
            y_mv_strings=out_motion["strings"][0],
            z_mv_strings=out_motion["strings"][1],
            y_strings=out_res["strings"][0],
            z_strings=out_res["strings"][1],
            mv_shape=out_motion["shape"],
            main_shape=out_res["shape"]
        )

    def decompress_P_frame(self, input: PFrameCompressOutput) -> PFrameDecompressOutput:
        # motion
        y_motion_hat = self.motion_hyperprior.decompress(
            input.mv_strings, 
            input.mv_shape
        )

        # decode the space-scale flow information
        motion_info = self.motion_decoder(y_motion_hat)
        x_pred = self.forward_prediction(input.refer_frame, motion_info)

        # residual
        y_res_hat = self.res_hyperprior.decompress(
            input.strings, 
            input.main_shape
        )

        # y_combine
        y_combine = torch.cat((y_res_hat, y_motion_hat), dim=1)
        x_res_hat = self.res_decoder(y_combine)

        # final reconstruction: prediction + residual
        x_rec = x_pred + x_res_hat

        return PFrameDecompressOutput(recon_frame=x_rec)

    @staticmethod
    def gaussian_volume(x, sigma: float, num_levels: int):
        """Efficient gaussian volume construction.

        From: "Generative Video Compression as Hierarchical Variational Inference",
        by Yang et al.
        """
        k = 2 * int(math.ceil(3 * sigma)) + 1
        device = x.device
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32

        kernel = gaussian_kernel2d(k, sigma, device=device, dtype=dtype)
        volume = [x.unsqueeze(2)]
        x = gaussian_blur(x, kernel=kernel)
        volume += [x.unsqueeze(2)]
        for i in range(1, num_levels):
            x = F.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
            x = gaussian_blur(x, kernel=kernel)
            interp = x
            for _ in range(0, i):
                interp = F.interpolate(
                    interp, scale_factor=2, mode="bilinear", align_corners=False
                )
            volume.append(interp.unsqueeze(2))
        return torch.cat(volume, dim=2)

    def warp_volume(self, volume, flow, scale_field, padding_mode: str = "border"):
        """3D volume warping."""
        if volume.ndimension() != 5:
            raise ValueError(
                f"Invalid number of dimensions for volume {volume.ndimension()}"
            )

        N, C, _, H, W = volume.size()

        with amp.autocast(device_type=volume.device.type, enabled=False):
            grid = meshgrid2d(N, C, H, W, volume.device)
            update_grid = grid + flow.permute(0, 2, 3, 1).float()
            update_scale = scale_field.permute(0, 2, 3, 1).float()
            volume_grid = torch.cat((update_grid, update_scale), dim=-1).unsqueeze(1)

            out = F.grid_sample(
                volume.float(),
                volume_grid,
                padding_mode=padding_mode,
                align_corners=False,
            )
        return out.squeeze(2)

    def forward_prediction(self, x_ref, motion_info):
        flow, scale_field = motion_info.chunk(2, dim=1)

        volume = self.gaussian_volume(x_ref, self.sigma0, self.num_levels)
        x_pred = self.warp_volume(volume, flow, scale_field)
        return x_pred