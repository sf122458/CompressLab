from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import torch, math
from typing import List, Tuple, Union, Dict
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

###########################################################
#                       PFrameCodec                       #
###########################################################

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


    # below metrics are used in DVC official implementation, not required in all codecs
    warp_frame: torch.Tensor = None
    prediction: torch.Tensor = None
    warp_loss: torch.Tensor = field(init=False, default=None)
    warp_psnr: float = field(init=False, default=None)
    inter_loss: torch.Tensor = field(init=False, default=None)
    inter_psnr: float = field(init=False, default=None)


    def __post_init__(self):
        N, _, H, W = self.input_frame.shape
        num_pixels = N * H * W
        if self.likelihoods is not None:
            self.bpp = self.likelihoods.calc_bpp(num_pixels)

        self.mse_loss = F.mse_loss(self.recon_frame, self.input_frame)
        self.psnr = 10 * torch.log10(1.0 / self.mse_loss).item()

        if self.warp_frame is not None:
            self.warp_loss = F.mse_loss(self.warp_frame, self.input_frame)
            self.warp_psnr = 10 * torch.log10(1.0 / self.warp_loss).item()
        if self.prediction is not None:
            self.inter_loss = F.mse_loss(self.prediction, self.input_frame)
            self.inter_psnr = 10 * torch.log10(1.0 / self.inter_loss).item()

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
class PFrameCodecCompressInput:
    """
    Input of the `PFrameCodec` 's `compress` method.

    Attributes:
        I_frame (torch.Tensor): The I-frame tensor.
        P_frames (List[torch.Tensor]): A list of P-frame tensors.

        I_frame_bpp (Union[float, torch.Tensor], optional): Bits per pixel for the I-frame.
        I_frame_psnr (Union[float, torch.Tensor], optional): Peak signal-to-noise ratio for the I-frame.
        I_frame_ms_ssim (Union[float, torch.Tensor], optional): Multi-scale structural similarity index for the I-frame.
    """
    I_frame: torch.Tensor
    P_frames: List[torch.Tensor]

    # In DVC implementation, the I frame is pre-compressed with H.265 codec so below attributes are required to store the pre-compressed I frame metrics.
    I_frame_bpp: Union[float, torch.Tensor] = None
    I_frame_psnr: Union[float, torch.Tensor] = None
    I_frame_ms_ssim: Union[float, torch.Tensor] = None


@dataclass
class PFrameCodecCompressOutput:
    """
    Output of the `PFrameCodec` 's `compress` method.

    Attributes:
        compress_input (PFrameCodecCompressInput): The input of `compress` method.
        P_frame_compress_output (List[PFrameCompressOutput]): A list of compressed P-frame outputs.

        bpp (float): Average bits per pixel for all frames.
        psnr (float, optional): Average peak signal-to-noise ratio for all frames.
        ms_ssim (float, optional): Average multi-scale structural similarity index for all frames.
    """
    compress_input: PFrameCodecCompressInput
    P_frame_compress_output: List[PFrameCompressOutput] = field(default_factory=list)
    
    # automatically calculated in `__post_init__`
    bpp: float = field(init=False)
    psnr: float = field(init=False, default=None)
    ms_ssim: float = field(init=False, default=None)

    def __post_init__(self):
        avg_bpp = self.compress_input.I_frame_bpp.item() if isinstance(self.compress_input.I_frame_bpp, torch.Tensor) \
            else self.compress_input.I_frame_bpp
        avg_psnr = self.compress_input.I_frame_psnr.item() if isinstance(self.compress_input.I_frame_psnr, torch.Tensor) \
            else self.compress_input.I_frame_psnr
        avg_ms_ssim = self.compress_input.I_frame_ms_ssim.item() if isinstance(self.compress_input.I_frame_ms_ssim, torch.Tensor) \
            else self.compress_input.I_frame_ms_ssim

        for output in self.P_frame_compress_output:
            avg_bpp += output.bpp
            avg_psnr += output.psnr
            avg_ms_ssim = output.ms_ssim
        
        self.bpp = avg_bpp / (len(self.P_frame_compress_output) + 1)
        self.psnr = avg_psnr / (len(self.P_frame_compress_output) + 1)
        self.ms_ssim = avg_ms_ssim / (len(self.P_frame_compress_output) + 1)



@dataclass
class PFrameCodecDecompressOutput:
    """
    Output of the `PFrameCodec` 's `decompress` method.
    
    Attributes:
        recon_frames (List[torch.Tensor]): A list of reconstructed P-frame tensors.
    """
    recon_frames: List[torch.Tensor]



class PFrameCodec(ABC):
    """
    Abstract base class for codecs that only encode one P-frame in one `forward` operation.

    """
    @abstractmethod
    def forward(self, input: PFrameForwardInput) -> PFrameForwardOutput:
        """
        Encode and decode the P-frame and obtain the metrics.
        """
        pass

    @abstractmethod
    def compress(self, input: PFrameCompressInput) -> PFrameCompressOutput:
        """
        Compress the P-frame sequences and obtain the compressed output.
        """
        pass

    @abstractmethod
    def decompress(self, input: PFrameCompressOutput) -> PFrameDecompressOutput:
        """
        Decompress the P-frame from the compressed output and obtain the reconstructed frame.
        """
        pass

    @abstractmethod
    def compress_P_frame(self, input: PFrameCompressInput) -> PFrameCompressOutput:
        """
        Compress a single P-frame and obtain the compressed output.
        """
        pass

    @abstractmethod
    def decompress_P_frame(self, input: PFrameCompressOutput) -> PFrameDecompressOutput:
        """
        Decompress a single P-frame from the compressed output and obtain the reconstructed frame.
        """
        pass


###########################################################
#                      IPFrameCodec                       #
###########################################################


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
        self.psnr = 10 * torch.log10(1.0 / self.mse_loss).item()


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
class IPFrameCodecForwardInput:
    """
    Input of the `IPFrameCodec` 's `forward` method.

    Attributes:
        I_frame (torch.Tensor): The I-frame tensor.
        P_frames (List[torch.Tensor]): A list of P-frame tensors.
    """
    I_frame: torch.Tensor
    P_frames: List[torch.Tensor]


@dataclass
class IPFrameCodecForwardOutput:
    """
    Output of the `IPFrameCodec` 's `forward` method.

    Attributes:
        out_list (List[Union[IFrameForwardOutput, PFrameForwardOutput]]): A list containing the outputs of the I-frame and P-frames.
        
        bpp (torch.Tensor): Total bits per pixel for all frames.
        mse_loss (torch.Tensor): Total mean squared error loss for all frames.
        psnr (float): Total peak signal-to-noise ratio for all frames.
    """
    out_list: List[Union[IFrameForwardOutput, PFrameForwardOutput]]

    bpp: torch.Tensor = field(init=False, default=0)
    mse_loss: torch.Tensor = field(init=False, default=0)
    psnr: float = field(init=False, default=0)

    def __post_init__(self):
        for out in self.out_list:
            self.bpp += out.bpp
            self.mse_loss += out.mse_loss
            self.psnr += out.psnr


@dataclass
class IPFrameCodecCompressInput:
    """
    Input of the `IPFrameCodec` 's `compress` method.

    Attributes:
        I_frame (torch.Tensor): The I-frame tensor.
        P_frames (List[torch.Tensor]): A list of P-frame tensors.
    """
    I_frame: torch.Tensor
    P_frames: List[torch.Tensor]


@dataclass
class IPFrameCodecCompressOutput:
    """
    Output of the `IPFrameCodec` 's `compress` method.

    Attributes:
        I_frame_compress_output (IFrameCompressOutput): The compressed output of the I-frame.
        P_frame_compress_output (List[PFrameCompressOutput]): A list of compressed outputs for P-frames.

        bpp (float): Total bits per pixel for all frames.
        psnr (float): Total peak signal-to-noise ratio for all frames.
        ms_ssim (float): Total multi-scale structural similarity index for all frames.
    """
    I_frame_compress_output: IFrameCompressOutput
    P_frame_compress_output: List[PFrameCompressOutput] = field(default_factory=list)

    # automatically calculated in `__post_init__`
    bpp: float = field(init=False)
    psnr: float = field(init=False, default=None)
    ms_ssim: float = field(init=False, default=None)

    def __post_init__(self):
        self.bpp = self.I_frame_compress_output.bpp
        self.psnr = self.I_frame_compress_output.psnr
        self.ms_ssim = self.I_frame_compress_output.ms_ssim

        for output in self.P_frame_compress_output:
            self.bpp += output.bpp
            self.psnr += output.psnr
            self.ms_ssim += output.ms_ssim


@dataclass
class IPFrameCodecDecompressOutput:
    """
    Output of the `IPFrameCodec` 's `decompress` method.

    Attributes:
        recon_frames (List[torch.Tensor]): A list of reconstructed frames, including the I-frame and P-frames.
    """
    recon_frames: List[torch.Tensor]


class IPFrameCodec(ABC):
    """
    Abstract base class for codecs that encode a single I-frame and multiple P-frames in one `forward` operation.
    """
    @abstractmethod
    def forward_I_frame(self, input: IFrameForwardInput) -> IFrameForwardOutput:
        """
        Encode and decode a single I-frame and obtain the metrics.
        """
        pass

    @abstractmethod
    def compress_I_frame(self, input: IFrameCompressInput) -> IFrameCompressOutput:
        """
        Compress a single I-frame and obtain the compressed data and metrics.
        """
        pass

    @abstractmethod
    def decompress_I_frame(self, input: IFrameCompressOutput) -> IFrameDecompressOutput:
        """
        Decompress a single I-frame from the compressed data and obtain the reconstructed frame.
        """
        pass

    @abstractmethod
    def forward_P_frame(self, input: PFrameForwardInput) -> PFrameForwardOutput:
        """
        Encode and decode a single P-frame and obtain the metrics.
        """
        pass

    @abstractmethod
    def compress_P_frame(self, input: PFrameCompressInput) -> PFrameCompressOutput:
        """
        Compress a single P-frame and obtain the compressed data and metrics.
        """
        pass

    @abstractmethod
    def decompress_P_frame(self, input: PFrameCompressOutput) -> PFrameDecompressOutput:
        """
        Decompress a single P-frame from the compressed data and obtain the reconstructed frame.
        """
        pass

    @abstractmethod
    def forward(self, input: IPFrameCodecForwardInput) -> IPFrameCodecForwardOutput:
        """
        Encodes and decodes the I-frame and P-frames, returning their metrics.
        """
        pass

    @abstractmethod
    def compress(self, input: IPFrameCodecCompressInput) -> IPFrameCodecCompressOutput:
        """
        Compresses the I-frame and P-frames, returning their compressed data and metrics.
        """
        pass

    @abstractmethod
    def decompress(self, input: IPFrameCodecCompressOutput) -> IPFrameCodecDecompressOutput:
        """
        Decompresses the I-frame and P-frames from the compressed data, returning the reconstructed frames.
        """
        pass


