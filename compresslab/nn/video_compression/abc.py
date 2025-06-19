from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import torch, math
from typing import List, Tuple, Union, Dict
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from compresslab.nn.video_compression.image_codec import load_model
from compresslab.core.models import CompressionModel

@dataclass
class PFrameLikelihoods:
    """
    A dataclass to hold the likelihoods of the P-frame.
    
    Attributes:
        y_mv (torch.Tensor): The likelihoods of the motion vectors.
        z_mv (torch.Tensor): The likelihoods of the motion vector prior.
        y (torch.Tensor): The likelihoods of the residual / contextual tensor.
        z (torch.Tensor): The likelihoods of the residual / contextual prior.

        bpp_y_mv (torch.Tensor): Bits per pixel for the motion vectors.
        bpp_z_mv (torch.Tensor): Bits per pixel for the motion vector prior.
        bpp_y (torch.Tensor): Bits per pixel for the residual / contextual tensor.
        bpp_z (torch.Tensor): Bits per pixel for the residual / contextual prior.
        bpp (torch.Tensor): Total bits per pixel for the P-frame likelihoods.
    """
    y_mv: torch.Tensor
    z_mv: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor

    # Below attributes will automatically calculated in `calc_bpp`.
    bpp_y_mv: torch.Tensor = field(init=False)
    bpp_z_mv: torch.Tensor = field(init=False)
    bpp_y: torch.Tensor = field(init=False)
    bpp_z: torch.Tensor = field(init=False)
    bpp: torch.Tensor = field(init=False)

    def calc_bpp(self, num_pixels: int):
        """
        Calculate the bits per pixel (bpp) for the likelihoods.
        
        Args:
            num_pixels (int): The number of pixels in the frame.
        
        Returns:
            None: The bpp values are updated in the instance variables.
        """
        self.bpp_y_mv = torch.log(self.y_mv).sum() / (-math.log(2) * num_pixels)
        self.bpp_z_mv = torch.log(self.z_mv).sum() / (-math.log(2) * num_pixels)
        self.bpp_y = torch.log(self.y).sum() / (-math.log(2) * num_pixels)
        self.bpp_z = torch.log(self.z).sum() / (-math.log(2) * num_pixels)
        self.bpp = self.bpp_y_mv + self.bpp_z_mv + self.bpp_y + self.bpp_z
        return self.bpp


@dataclass
class PFrameForwardInput:
    """
    Input of the `forward_P_frame` method.
    
    Attributes:
        input_frame (torch.Tensor): The input frame tensor.
        refer_frame (torch.Tensor): The reference frame tensor.
    """
    input_frame: torch.Tensor
    refer_frame: torch.Tensor


@dataclass
class PFrameForwardOutput:
    """
    Output of the `forward_P_frame` method.

    Attributes:
        input_frame (torch.Tensor): The input frame tensor. Used for calculating psnr.
        recon_frame (torch.Tensor): The reconstructed frame tensor.
        likelihoods (PFrameLikelihoods): The likelihoods of the P-frame.
        
        bpp (torch.Tensor): Bits per pixel for the P-frame.
        mse_loss (torch.Tensor): Mean squared error loss between the input and reconstructed frames.
        psnr (float): Peak signal-to-noise ratio of the reconstructed frame.
        
        warp_frame (torch.Tensor, optional): The warped frame tensor, if applicable.
        prediction (torch.Tensor, optional): The predicted frame tensor, if applicable.
        warp_loss (torch.Tensor, optional): MSE loss for the warped frame.
        warp_psnr (float, optional): PSNR for the warped frame.
        inter_loss (torch.Tensor, optional): MSE loss for the predicted frame.
        inter_psnr (float, optional): PSNR for the predicted frame.
    """
    input_frame: torch.Tensor
    recon_frame: torch.Tensor
    likelihoods: PFrameLikelihoods

    # automatically calculated in `__post_init__`
    bpp: torch.Tensor = field(init=False, default=None)
    mse_loss: torch.Tensor = field(init=False)
    psnr: float = field(init=False)

    def __post_init__(self):
        N, _, H, W = self.input_frame.shape
        num_pixels = N * H * W
        if self.likelihoods is not None:
            self.bpp = self.likelihoods.calc_bpp(num_pixels)

        self.mse_loss = F.mse_loss(self.recon_frame, self.input_frame)
        self.psnr = 10 * torch.log10(1.0 / self.mse_loss).detach()


@dataclass
class PFrameCompressInput:
    """
    Input of the `compress_P_frame` method.
    
    Attributes:
        input_frame (torch.Tensor): The input frame tensor.
        refer_frame (torch.Tensor): The reference frame tensor.
    """
    input_frame: torch.Tensor
    refer_frame: torch.Tensor


@dataclass
class PFrameCompressOutput:
    """
    Output of the `compress_P_frame` method.

    Attributes:
        input_frame (torch.Tensor): The input frame tensor. Required for calculating psnr.
        recon_frame (torch.Tensor): The reconstructed frame tensor.
        refer_frame (torch.Tensor): The reference frame tensor.
        y_mv_strings (List[bytes]): Compressed motion vector strings.
        y_strings (List[bytes]): Compressed residual / contextual strings.
        z_mv_strings (List[bytes], optional): Compressed hyper motion vector strings.
        z_strings (List[bytes], optional): Compressed hyper residual / contextual strings.
        mv_shape (Tuple[int, int]): Shape of the motion vectors.
        main_shape (Tuple[int, int]): Shape of the residual / contextual content.

        bpp (float): Bits per pixel for all compressed strings.
        psnr (float, optional): Peak signal-to-noise ratio of the reconstructed frame.
        ms_ssim (float, optional): Multi-scale structural similarity index of the reconstructed frame.

        mv_strings (List[List[bytes]]): Compatible with CompressAI, contains motion vector strings.
        strings (List[List[bytes]]): Compatible with CompressAI, contains residual strings.
    """
    input_frame: torch.Tensor
    recon_frame: torch.Tensor
    refer_frame: torch.Tensor
    y_mv_strings: List[bytes]
    y_strings: List[bytes]
    z_mv_strings: List[bytes] = None
    z_strings: List[bytes] = None
    mv_shape: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    main_shape: Tuple[int, int] = field(default_factory=lambda: (0, 0))

    # automatically calculated in `__post_init__`
    bpp: float = field(init=False)
    psnr: float = field(init=False, default=None)
    ms_ssim: float = field(init=False, default=None)

    # compatible with CompressAI
    mv_strings: List[List[bytes]] = field(init=False)
    strings: List[List[bytes]] = field(init=False)

    def __post_init__(self):
        N, _, H, W = self.recon_frame.shape
        num_pixels = N * H * W
        total_bits = sum(len(s) * 8 for s in self.y_mv_strings + self.z_mv_strings + self.y_strings + self.z_strings)
        self.bpp = total_bits / num_pixels

        mse_loss = F.mse_loss(self.recon_frame, self.input_frame)
        self.psnr = 10 * torch.log10(1.0 / mse_loss).item()
        self.ms_ssim = ms_ssim(self.recon_frame, self.input_frame, data_range=1.0, size_average=True).item()

        self.mv_strings = [self.y_mv_strings, self.z_mv_strings] if self.z_mv_strings else [self.y_mv_strings]
        self.strings = [self.y_strings, self.z_strings] if self.z_strings else [self.y_strings]
        

@dataclass
class PFrameDecompressOutput:
    """
    Output of the `decompress_P_frame` method.

    Attributes:
        recon_frame (torch.Tensor): The reconstructed P frame tensor.
    """
    recon_frame: torch.Tensor


@dataclass
class IFrameLikelihoods:
    """
    A dataclass to hold the likelihoods of the I-frame.

    Attributes:
        y (torch.Tensor): The likelihoods of the factorized-prior tensor.
        z (torch.Tensor): The likelihoods of the hyper-prior tensor.

        bpp_y (torch.Tensor): Bits per pixel for the factorized-prior tensor.
        bpp_z (torch.Tensor): Bits per pixel for the hyper-prior tensor.
        bpp (torch.Tensor): Total bits per pixel for the I-frame likelihoods.
    """
    y: torch.Tensor
    z: torch.Tensor

    # Below attributes will automatically calculated in `calc_bpp`.
    bpp_y: torch.Tensor = field(init=False)
    bpp_z: torch.Tensor = field(init=False)
    bpp: torch.Tensor = field(init=False)

    def calc_bpp(self, num_pixels: int):
        self.bpp_y = torch.log(self.y).sum() / (-math.log(2) * num_pixels)
        self.bpp_z = torch.log(self.z).sum() / (-math.log(2) * num_pixels)
        self.bpp = self.bpp_y + self.bpp_z
        return self.bpp


@dataclass
class IFrameForwardInput:
    """
    Input of the `IPFrameCodec` 's `forward_I_frame` method.

    Attributes:
        input_frame (torch.Tensor): The input frame tensor.
    """
    input_frame: torch.Tensor


@dataclass
class IFrameForwardOutput:
    """
    Output of the `IPFrameCodec` 's `forward_I_frame` method.

    Attributes:
        input_frame (torch.Tensor): The input frame tensor. Used for calculating psnr.
        recon_frame (torch.Tensor): The reconstructed frame tensor.
        likelihoods (IFrameLikelihoods): The likelihoods of the I-frame.

        bpp (torch.Tensor): Bits per pixel for the I-frame.
        mse_loss (torch.Tensor): Mean squared error loss between the input and reconstructed frames.
        psnr (float): Peak signal-to-noise ratio of the reconstructed frame. 
    """
    input_frame: torch.Tensor
    recon_frame: torch.Tensor
    likelihoods: IFrameLikelihoods

    # automatically calculated in `__post_init__`
    bpp: torch.Tensor = field(init=False)
    mse_loss: torch.Tensor = field(init=False)
    psnr: float = field(init=False)

    def __post_init__(self):
        N, _, H, W = self.input_frame.shape
        num_pixels = N * H * W
        if self.likelihoods is not None:
            self.bpp = self.likelihoods.calc_bpp(num_pixels)
        self.mse_loss = F.mse_loss(self.recon_frame, self.input_frame)
        self.psnr = 10 * torch.log10(1.0 / self.mse_loss).detach()


@dataclass
class IFrameCompressInput:
    """
    Input of the `IPFrameCodec` 's `compress_I_frame` method.

    Attributes:
        input_frame (torch.Tensor): The input frame tensor.
    """
    input_frame: torch.Tensor


@dataclass
class IFrameCompressOutput:
    """
    Output of the `IPFrameCodec` 's `compress_I_frame` method.

    Attributes:
        input_frame (torch.Tensor): The input frame tensor. Used for calculating psnr.
        recon_frame (torch.Tensor): The reconstructed frame tensor.
        y_strings (List[bytes]): Compressed factorized-prior strings.
        z_strings (List[bytes], optional): Compressed hyper-prior strings.
        shape (Tuple[int, int]): Shape of the hyper-prior tensor.

        bpp (float): Bits per pixel for all compressed strings.
        psnr (float, optional): Peak signal-to-noise ratio of the reconstructed frame.
        ms_ssim (float, optional): Multi-scale structural similarity index of the reconstructed frame.

        strings (List[List[bytes]]): Compatible with CompressAI, contains compressed strings.
    """
    input_frame: torch.Tensor
    recon_frame: torch.Tensor
    y_strings: List[bytes]
    z_strings: List[bytes] = field(default_factory=list)
    shape: Tuple[int, int] = field(default_factory=lambda: (0, 0))

    # automatically calculated in `__post_init__`
    bpp: float = field(init=False)
    psnr: float = field(init=False, default=None)
    ms_ssim: float = field(init=False, default=None)

    # compatible with CompressAI
    strings: List[List[bytes]] = field(init=False)
    

    def __post_init__(self):
        N, _, H, W = self.input_frame.shape
        num_pixels = N * H * W
        total_bits = sum(len(s) * 8 for s in self.y_strings + self.z_strings)

        self.bpp = total_bits / num_pixels
        mse_loss = F.mse_loss(self.recon_frame, self.input_frame)
        self.psnr = 10 * torch.log10(1.0 / mse_loss).item()
        self.ms_ssim = ms_ssim(self.recon_frame, self.input_frame, data_range=1.0, size_average=True).item()

        self.strings = [self.y_strings, self.z_strings]


@dataclass
class IFrameDecompressOutput:
    """
    Output of the `IPFrameCodec` 's `decompress_I_frame` method.

    Attributes:
        recon_frame (torch.Tensor): The reconstructed I frame tensor.
    """
    recon_frame: torch.Tensor


@dataclass
class VideoCodecForwardInput:
    frames: List[torch.Tensor] = None
    I_frame: torch.Tensor = None
    P_frames: List[torch.Tensor] = None


    def __post_init__(self):
        if self.frames is not None:
            assert len(self.frames) > 1
            self.I_frame = self.frames[0]
            self.P_frames = self.frames[1:]
        
        elif self.I_frame is not None and self.P_frames is not None:
            assert isinstance(self.I_frame, torch.Tensor)
            assert isinstance(self.P_frames, list) and all(isinstance(frame, torch.Tensor) for frame in self.P_frames)
            self.frames = [self.I_frame] + self.P_frames

        else:
            raise ValueError("Either 'frames' or both 'I_frame' and 'P_frames' must be provided.")
        
    def to_dict(self):
        return {
            "frames": self.frames,
            "I_frame": self.I_frame,
            "P_frames": self.P_frames
        }


@dataclass
class VideoCodecForwardOutput:
    I_frame_output: IFrameForwardOutput = None
    P_frame_output: List[PFrameForwardOutput] = field(default_factory=list)

    # automatically calculated in `__post_init__`
    I_frame_bpp: float = field(init=False, default=0)
    I_frame_psnr: float = field(init=False, default=0)

    P_frame_bpp: float = field(init=False, default=0)
    P_frame_psnr: float = field(init=False, default=0)

    bpp: torch.Tensor = field(init=False, default=0)
    mse_loss: torch.Tensor = field(init=False, default=0)
    psnr: float = field(init=False, default=0)

    def __post_init__(self):
        self.I_frame_bpp = self.I_frame_output.bpp
        self.I_frame_psnr = self.I_frame_output.psnr

        total_bpp, total_psnr, total_mse_loss = 0, 0, 0

        for output in self.P_frame_output:
            total_bpp += output.bpp
            total_mse_loss += output.mse_loss
            total_psnr += output.psnr

        self.P_frame_bpp = total_bpp / len(self.P_frame_output)
        self.P_frame_psnr = total_psnr / len(self.P_frame_output)

        total_bpp += self.I_frame_bpp
        total_psnr += self.I_frame_psnr
        total_mse_loss += self.I_frame_output.mse_loss

        self.bpp = total_bpp / (len(self.P_frame_output) + 1)
        self.mse_loss = total_mse_loss / (len(self.P_frame_output) + 1)
        self.psnr = total_psnr / (len(self.P_frame_output) + 1)

    def to_dict(self):  #NOTE: vmap support
        return {
            "bpp": self.bpp,
            "mse_loss": self.mse_loss,
            "psnr": self.psnr
        }


@dataclass
class VideoCodecCompressInput:
    frames: List[torch.Tensor] = None
    I_frame: torch.Tensor = None
    P_frames: List[torch.Tensor] = None

    def __post_init__(self):
        if self.frames is not None:
            assert len(self.frames) > 1
            self.I_frame = self.frames[0]
            self.P_frames = self.frames[1:]
        
        elif self.I_frame is not None and self.P_frames is not None:
            assert isinstance(self.I_frame, torch.Tensor)
            assert isinstance(self.P_frames, list) and all(isinstance(frame, torch.Tensor) for frame in self.P_frames)
            self.frames = [self.I_frame] + self.P_frames

        else:
            raise ValueError("Either 'frames' or both 'I_frame' and 'P_frames' must be provided.")
    

@dataclass
class VideoCodecCompressOutput:
    frame_output: List[Union[IFrameCompressOutput, PFrameCompressOutput]] = None
    I_frame_output: IFrameCompressOutput = None
    P_frame_output: List[PFrameCompressOutput] = None

    # automatically calculated in `__post_init__`
    I_frame_bpp: float = field(init=False, default=0)
    I_frame_psnr: float = field(init=False, default=0)
    I_frame_ms_ssim: float = field(init=False, default=0)

    P_frame_bpp: float = field(init=False, default=0)
    P_frame_psnr: float = field(init=False, default=0)
    P_frame_ms_ssim: float = field(init=False, default=0)

    bpp: float = field(init=False, default=0)
    psnr: float = field(init=False, default=0)
    ms_ssim: float = field(init=False, default=0)

    def __post_init__(self):
        if self.frame_output is not None:
            self.I_frame_output = self.frame_output[0]
            self.P_frame_output = self.frame_output[1:]
        else:
            assert self.I_frame_output is not None
            assert isinstance(self.P_frame_output, list) and all(isinstance(output, PFrameCompressOutput) for output in self.P_frame_output)

        self.I_frame_bpp = self.I_frame_output.bpp
        self.I_frame_psnr = self.I_frame_output.psnr
        self.I_frame_ms_ssim = self.I_frame_output.ms_ssim

        total_bpp, total_psnr, total_ms_ssim = 0, 0, 0

        for output in self.P_frame_output:
            total_bpp += output.bpp
            total_psnr += output.psnr
            total_ms_ssim += output.ms_ssim

        self.P_frame_bpp = total_bpp / len(self.P_frame_output)
        self.P_frame_psnr = total_psnr / len(self.P_frame_output)
        self.P_frame_ms_ssim = total_ms_ssim / len(self.P_frame_output)

        total_bpp += self.I_frame_bpp
        total_psnr += self.I_frame_psnr
        total_ms_ssim += self.I_frame_ms_ssim        

        self.bpp = total_bpp / (len(self.P_frame_output) + 1)
        self.psnr = total_psnr / (len(self.P_frame_output) + 1)
        self.ms_ssim = total_ms_ssim / (len(self.P_frame_output) + 1)


@dataclass
class VideoCodecDecompressOutput:
    recon_frames: List[torch.Tensor]


class VideoCodec(CompressionModel, ABC):
    """
    Video codec abstract base class.

    TODO: If the model does not support I-frame encoding
    """
    def __init_subclass__(cls):
        # decorate the forward method for vmap support
        def vmap_forward(func):
            def wrapper(self, input):
                if isinstance(input, VideoCodecForwardInput):
                    return func(self, input)
                else:
                    # Assuming input is a list of tensors for vmap support
                    output = func(self, VideoCodecForwardInput(I_frame=input[0], P_frames=input[1:]))
                    return output.to_dict()
            return wrapper

        if hasattr(cls, 'forward') and callable(cls.forward):
            cls.forward = vmap_forward(cls.forward)
        

    def preprocess(self, codec: str=None, quality="low", **kwargs):
        self.codec = codec
        if codec is not None:
            self.image_codec = load_model(codec, quality=quality, **kwargs)


    def forward(self, input: Union[VideoCodecForwardInput, List[torch.Tensor]]) -> Union[VideoCodecForwardOutput, Dict[str, Union[float, torch.Tensor]]]:
        I_frame_out = self.forward_I_frame(
            IFrameForwardInput(
                input_frame=input.I_frame
            )
        )

        P_frame_out_list = []

        for i in range(len(input.P_frames)):
            P_frame_out = self.forward_P_frame(
                PFrameForwardInput(
                    input_frame=input.P_frames[i],
                    refer_frame=I_frame_out.recon_frame.detach() if i == 0 else P_frame_out.recon_frame
                )
            )

            P_frame_out_list.append(P_frame_out)

        return VideoCodecForwardOutput(
            I_frame_output=I_frame_out,
            P_frame_output=P_frame_out_list
        )

    def compress(self, input: VideoCodecCompressInput) -> VideoCodecCompressOutput:
        I_frame_out = self.compress_I_frame(
            IFrameCompressInput(
                input_frame=input.I_frame
            )
        )

        P_frame_out_list = []

        for i in range(len(input.P_frames)):
            P_frame_out = self.compress_P_frame(
                PFrameCompressInput(
                    input_frame=input.P_frames[i],
                    refer_frame=I_frame_out.recon_frame if i == 0 else P_frame_out.recon_frame
                )
            )

            P_frame_out_list.append(P_frame_out)

        return VideoCodecCompressOutput(
            I_frame_output=I_frame_out,
            P_frame_output=P_frame_out_list
        )
        

    def decompress(self, input: VideoCodecCompressOutput) -> VideoCodecDecompressOutput:
        dec_frames = []

        I_frame_out = self.decompress_I_frame(input.I_frame_output)
        dec_frames.append(I_frame_out.recon_frame)

        for i in range(len(input.P_frame_output)):
            P_frame_out = self.decompress_P_frame(input.P_frame_output[i])
            dec_frames.append(P_frame_out.recon_frame)

        return VideoCodecDecompressOutput(recon_frames=dec_frames)


    def forward_I_frame(self, input: IFrameForwardInput) -> IFrameForwardOutput:
        if self.codec is None:
            raise NotImplementedError("The I frame codec is not implemented.")
        else:
            out = self.image_codec(input.input_frame)
            return IFrameForwardOutput(
                input_frame=input.input_frame,
                recon_frame=out["x_hat"],
                likelihoods=IFrameLikelihoods(
                    y=out["likelihoods"]["y"],
                    z=out["likelihoods"]["z"]
                )
            )

    def compress_I_frame(self, input: IFrameCompressInput) -> IFrameCompressOutput:
        if self.codec is None:
            raise NotImplementedError("The I frame codec is not implemented.")
        else:
            out = self.image_codec.compress(input.input_frame)
            return IFrameCompressOutput(
                input_frame=input.input_frame,
                recon_frame=out["x_hat"],
                y_strings=out["strings"][0],
                z_strings=out["strings"][1],
                shape=out["shape"]
            )

    def decompress_I_frame(self, input: IFrameCompressOutput) -> IFrameDecompressOutput:
        if self.codec is None:
            raise NotImplementedError("The I frame codec is not implemented.")
        else:
            out = self.image_codec.decompress(input.strings, input.shape)
            return IFrameDecompressOutput(recon_frame=out["x_hat"])

    @abstractmethod
    def forward_P_frame(self, input: PFrameForwardInput) -> PFrameForwardOutput:
        pass


    @abstractmethod
    def compress_P_frame(self, input: PFrameCompressInput) -> PFrameCompressOutput:
        pass

    @abstractmethod
    def decompress_P_frame(self, input: PFrameCompressOutput) -> PFrameDecompressOutput:
        pass
