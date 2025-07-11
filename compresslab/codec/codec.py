"""
Modify from `compressai/utils/bench/codec.py`
"""

from abc import ABC, abstractmethod
import platform, os, io, sys
from typing import List, Union, Dict, Any
import logging
from PIL import Image
import numpy as np
import time
from compresslab.utils.logger import MetricLogger
import torch
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from tempfile import mkstemp
from .utils import *

class Codec(ABC):
    def __init__(
            self, 
            test_data_dir: str, 
            quality: Union[int, List[int]], 
            save_dir: str,
            *args, **kwargs):
        """
        Args:
            data_dir (str): Directory containing the images to be processed.
            quality (Union[int, List[int]]): Quality level(s) for the codec.
            save_dir (str): Directory where the metrics will be saved.
        """
        self.data_dir = test_data_dir
        self.quality = quality if isinstance(quality, list) else [quality]
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.temp_dir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), self.save_dir)

    def _load_img(self, img_path: str) -> np.array:
        return read_image(os.path.abspath(img_path))
    
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def _run_impl(self, img_path, quality, *args, **kwargs) -> Dict[str, Any]:
        """
        Args:
            img_path (str): Path to the image file.
            quality (int): Quality level for the codec.
        Returns:
            Dict[str, Any]: A dictionary containing the metrics for the codec.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def run(self):
        self.logger = MetricLogger(self.save_dir)

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "{task.completed}/{task.total}",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),

        ) as progress:
            task1 = progress.add_task(f"[white]Codec: {self.name}", total=len(self.quality))
            task2 = progress.add_task(f"[white]Level: ...", total=len(os.listdir(self.data_dir)))
            for quality in self.quality:
                progress.update(task1, description=f"[white]Codec: {self.name}")
                progress.update(task2, completed=0, description=f"[white]Level: {quality}")
                for filename in sorted(os.listdir(self.data_dir)):
                    if not filename.lower().endswith(tuple(IMG_EXTENSIONS)):
                        continue

                    img_path = os.path.join(self.data_dir, filename)
                    metrics = self._run_impl(img_path, quality)

                    self.logger.log(
                        f"codec_q{quality}",
                        metrics
                    )

                    progress.advance(task2, advance=1)
                progress.advance(task1, advance=1)
        
        self.logger.save()



class PillowCodec(Codec):

    fmt = None

    def _run_impl(self, img_path: str, quality: int) -> Dict[str, Any]:
        img = self._load_img(img_path)
        start = time.time()
        tmp = io.BytesIO()
        img.save(tmp, format=self.fmt, quality=quality)
        enc_time = time.time() - start
        tmp.seek(0)
        bits = tmp.getbuffer().nbytes * 8  # Convert bytes to bits

        start = time.time()
        rec = Image.open(tmp)
        rec.load()
        dec_time = time.time() - start

        psnr, ms_ssim = compute_metrics(img, rec)

        return {
            "bpp": float(bits) / (img.size[0] * img.size[1]),
            "psnr": psnr,
            "ms-ssim": ms_ssim,
            "encoding_time": enc_time,
            "decoding_time": dec_time,
        }



class JPEG(PillowCodec):

    fmt = "jpeg"
    
    @property
    def name(self) -> str:
        return "JPEG"

class WebP(PillowCodec):
    
    fmt = "webp"

    @property
    def name(self) -> str:
        return "WebP"

class BinaryCodec(Codec):

    fmt = None
    
    def _run_impl(self, img_path: str, quality: int) -> Dict[str, Any]:
        fd0, png_filepath = mkstemp(suffix=".png", dir=self.temp_dir)
        fd1, out_filepath = mkstemp(suffix=self.fmt, dir=self.temp_dir)

        # Encode
        start = time.time()
        run_command(self._get_encode_cmd(img_path, quality, out_filepath))
        enc_time = time.time() - start
        size = filesize(out_filepath)

        # Decode
        start = time.time()
        run_command(self._get_decode_cmd(out_filepath, png_filepath))
        dec_time = time.time() - start

        # Read image
        rec = read_image(png_filepath)
        os.close(fd0)
        os.remove(png_filepath)
        os.close(fd1)
        os.remove(out_filepath)

        img = self._load_img(img_path)
        bpp_val = float(size) * 8 / (img.size[0] * img.size[1])

        psnr, ms_ssim = compute_metrics(img, rec)

        return {
            "bpp": bpp_val,
            "psnr": psnr,
            "ms-ssim": ms_ssim,
            "encoding_time": enc_time,
            "decoding_time": dec_time,
        }


    def _get_encode_cmd(self, in_filepath, quality, out_filepath):
        raise NotImplementedError()

    def _get_decode_cmd(self, out_filepath, rec_filepath):
        raise NotImplementedError()

class JPEG2000(BinaryCodec):
    
    fmt = ".jp2"

    @property
    def name(self) -> str:
        return "JPEG2000"
    
    @property
    def description(self) -> str:
        return f"JPEG 2000. ffmpeg version {get_ffmpeg_version()}"
    

    def _get_encode_cmd(self, in_filepath, quality, out_filepath):
        cmd = [
            "ffmpeg",
            "-loglevel",
            "panic",
            "-y",
            "-i",
            in_filepath,
            "-vcodec",
            "jpeg2000",
            "-pix_fmt",
            "yuv444p",
            "-c:v",
            "libopenjpeg",
            "-compression_level",
            quality,
            out_filepath,
        ]
        return cmd

    def _get_decode_cmd(self, out_filepath, rec_filepath):
        cmd = ["ffmpeg", "-loglevel", "panic", "-y", "-i", out_filepath, rec_filepath]
        return cmd
    

class BPG(BinaryCodec):
    "BPG from Fabrice Bellard"

    fmt = ".bpg"

    @property
    def name(self) -> str:
        return (
            f"BPG {self.bitdepth}b {self.subsampling_mode} {self.encoder}"
            f"{self.color_mode}"
        )
    
    @property
    def description(self) -> str:
        return f"BPG. Version {get_bpg_version(self.encoder_path)}"

    def __init__(
            self, 
            encoder_path: str = "bpgenc",
            decoder_path: str = "bpgdec",
            subsampling_mode: str = "444", 
            bitdepth: int = 8,
            color_mode: str = "ycbcr",
            encoder: str = "x265",
        ):
        """
        Args:
            encoder_path (str): Path to the BPG encoder executable.
            decoder_path (str): Path to the BPG decoder executable.
            subsampling_mode (str): Chroma subsampling mode, either "444" or "420".
            bitdepth (int): Bit depth of the image, either 8 or 10.
            color_mode (str): Color mode, either "ycbcr" or "rgb".
            encoder (str): Encoder to use, either "x265" or "jctvc
        """

        assert subsampling_mode in ["444", "420"]
        assert bitdepth in [8, 10]
        assert color_mode in ["ycbcr", "rgb"]
        assert encoder in ["x265", "jctvc"]

        self.subsampling_mode = subsampling_mode
        self.bitdepth = bitdepth
        self.color_mode = color_mode
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.encoder = encoder

    def _get_encode_cmd(self, img_path: str, quality: int, out_filepath):
        if not 0 <= int(quality) <= 51:
            raise ValueError(f"Quality {quality} is out of range [0, 51].")
        
        cmd = [
            self.encoder_path,
            "-o",
            out_filepath,
            "-q",
            str(quality),
            "-f",
            self.subsampling_mode,
            "-e",
            self.encoder,
            "-c",
            self.color_mode,
            "-b",
            self.bitdepth,
            img_path
        ]
        return cmd
    
    def _get_decode_cmd(self, out_filepath: str, rec_filepath: str):
        cmd = [self.decoder_path, "-o", rec_filepath, out_filepath]
        return cmd
        

class HM(Codec):
    """HM: HEVC reference software"""
    
    fmt = ".bin"

    @property
    def name(self) -> str:
        return "HM"

    def __init__(self, config: str, build_dir: str = HM_BUILD_DIR, rgb: bool = False, **kwargs):
        """
        Args:
            build_dir (str): Directory containing the HM encoder and decoder executables.
            config (str): Path to the configuration file for the HM encoder.
            quality (Union[int, List[int]]): Quality level(s) for the codec.
            rgb (bool): Whether to use RGB color space. Default is False (YUV).
        """
        super().__init__(**kwargs)
        self.encoder_path = os.path.join(build_dir, "TAppEncoderStatic")
        self.decoder_path = os.path.join(build_dir, "TAppDecoderStatic")
        self.config_path = os.path.join(build_dir, "../cfg", config)
        self.rgb = rgb

    def _run_impl(self, img_path: str, quality: int) -> Dict[str, Any]:
        if not 0 <= quality <= 51:
            logging.warning(f"Quality {quality} is out of range [0, 51]. Skipping.")
            return {}
        
        bitdepth = 8

        # Convert input image to yuv 444 file
        img = self._load_img(img_path)
        arr = np.asarray(img, dtype=np.uint8)
        fd, yuv_path = mkstemp(suffix=".yuv", dir=self.temp_dir)
        out_filepath = os.path.splitext(yuv_path)[0] + self.fmt

        arr = arr.transpose((2, 0, 1))  # color channel first

        if not self.rgb:
            # convert rgb content to YCbCr
            rgb = torch.from_numpy(arr.copy()).float() / (2 ** bitdepth - 1)
            arr = np.clip(rgb2ycbcr(rgb).numpy(), 0, 1)
            arr = (arr * (2 ** bitdepth - 1)).astype(np.uint8)
        
        with open(yuv_path, "wb") as f:
            f.write(arr.tobytes())

        # Encode
        height, width = arr.shape[1:]

        cmd = [
            self.encoder_path,
            "-i",
            yuv_path,
            "-c",
            self.config_path,
            "-q",
            quality,
            "-o",
            "/dev/null",
            "-b",
            out_filepath,
            "-wdt",
            width,
            "-hgt",
            height,
            "-fr",
            "1",
            "-f",
            "1",
            "--InputChromaFormat=444",
            "--InputBitDepth=8",
            "--SEIDecodedPictureHash",
            "--Level=5.1",
            # "--CUNoSplitIntraACT=0",
            "--ConformanceWindowMode=1",
        ]

        if self.rgb:
            cmd += [
                "--InputColourSpaceConvert=RGBtoGBR",
                "--SNRInternalColourSpace=1",
                "--OutputInternalColourSpace=0",
            ]

        start = time.time()
        run_command(cmd)
        enc_time = time.time() - start

        # cleanup encoder input
        os.close(fd)
        os.unlink(yuv_path)

        # Decode
        cmd = [self.decoder_path, "-b", out_filepath, "-o", yuv_path, "-d", "8"]

        if self.rgb:
            cmd += ["--OutputColourSpaceConvert=GBRtoRGB"]

        start = time.time()
        run_command(cmd)
        dec_time = time.time() - start

        # compute metrics
        rec_arr = np.fromfile(yuv_path, dtype=np.uint8)
        rec_arr = rec_arr.reshape(arr.shape)
        arr = arr.astype(np.float32) / (2 ** bitdepth - 1)
        rec_arr = rec_arr.astype(np.float32) / (2 ** bitdepth - 1)
        if not self.rgb:
            arr = ycbcr2rgb(torch.from_numpy(arr.copy())).numpy()
            rec_arr = ycbcr2rgb(torch.from_numpy(rec_arr.copy())).numpy()

        bpp = filesize(out_filepath) * 8.0 / (arr.shape[1] * arr.shape[2])
        # cleanup
        os.unlink(yuv_path)
        os.unlink(out_filepath)

        rec = Image.fromarray(
            (rec_arr.clip(0, 1).transpose(1, 2, 0) * (2 ** bitdepth - 1)).astype(np.uint8)
        )

        psnr, ms_ssim = compute_metrics(img, rec)

        return {
            "bpp": bpp,
            "psnr": psnr,
            "ms-ssim": ms_ssim,
            "encoding_time": enc_time,
            "decoding_time": dec_time,
        }


class VTM(Codec):
    """VTM: VVC reference software"""

    fmt = ".bin"

    @property
    def name(self) -> str:
        return "VTM"

    def __init__(self, config: str, build_dir: str = VTM_BUILD_DIR, rgb: bool = False, **kwargs):
        """
        Args:
            build_dir (str): Directory containing the VTM encoder and decoder executables.
            config (str): Path to the configuration file for the VTM encoder.
            rgb (bool): Whether to use RGB color space. Default is False (YUV).
        """
        super().__init__(**kwargs)
        self.encoder_path = self.get_encoder_path(build_dir)
        self.decoder_path = self.get_decoder_path(build_dir)
        self.config_path = os.path.join(build_dir, "../cfg", config)
        self.rgb = rgb

    def _run_impl(self, img_path: str, quality: int) -> Dict[str, Any]:
        if not 0 <= quality <= 63:
            logging.warning(f"Quality {quality} is out of range [0, 63]. Skipping.")
            return {}

        bitdepth = 8

        img = self._load_img(img_path)

        arr = np.asarray(img, dtype=np.uint8)
        fd, yuv_path = mkstemp(suffix=".yuv", dir=self.temp_dir)
        out_filepath = os.path.splitext(yuv_path)[0] + self.fmt

        arr = arr.transpose((2, 0, 1)) # color channel first

        if not self.rgb:
            # convert rgb content to YCbCr
            rgb = torch.from_numpy(arr.copy()).float() / (2 ** bitdepth - 1)
            arr = np.clip(rgb2ycbcr(rgb).numpy(), 0, 1)
            arr = (arr * (2 ** bitdepth - 1)).astype(np.uint8)

        with open(yuv_path, "wb") as f:
            f.write(arr.tobytes())

        # Encode
        height, width = arr.shape[1:]
        cmd = [
            self.encoder_path,
            "-i",
            yuv_path,
            "-c",
            self.config_path,
            "-q",
            quality,
            "-o",
            "/dev/null",
            "-b",
            out_filepath,
            "-wdt",
            width,
            "-hgt",
            height,
            "-fr",
            "1",
            "-f",
            "1",
            "--InputChromaFormat=444",
            "--InputBitDepth=8",
            "--ConformanceWindowMode=1"
        ]

        if self.rgb:
            cmd += [
                "--InputColourSpaceConvert=RGBtoGBR",
                "--SNRInternalColourSpace=1",
                "--OutputInternalColourSpace=0",
            ]

        start = time.time()
        run_command(cmd)
        enc_time = time.time() - start

        # cleanup encoder input
        os.close(fd)
        os.unlink(yuv_path)

        # Decode
        cmd = [self.decoder_path, "-b", out_filepath, "-o", yuv_path, "-d", 8]
        if self.rgb:
            # cmd += ["--OutputInternalColourSpace=GBRtoRGB"]
            cmd += ["--OutputColourSpaceConvert=GBRtoRGB"]  # support version: VTM Decoder Version 23.10

        start = time.time()
        run_command(cmd)
        dec_time = time.time() - start

        # compute metrics
        rec_arr = np.fromfile(yuv_path, dtype=np.uint8)
        rec_arr = rec_arr.reshape(arr.shape)

        arr = arr.astype(np.float32) / (2 ** bitdepth - 1)
        rec_arr = rec_arr.astype(np.float32) / (2 ** bitdepth - 1)

        if not self.rgb:
            arr = ycbcr2rgb(torch.from_numpy(arr.copy())).numpy()
            rec_arr = ycbcr2rgb(torch.from_numpy(rec_arr.copy())).numpy()

        bpp = filesize(out_filepath) * 8.0 / (arr.shape[1] * arr.shape[2])

        # cleanup
        os.unlink(yuv_path)
        os.unlink(out_filepath)

        rec = Image.fromarray(
            (rec_arr.clip(0, 1).transpose(1, 2, 0) * (2 ** bitdepth - 1)).astype(np.uint8)
        )

        psnr, ms_ssim = compute_metrics(img, rec)

        return {
            "bpp": bpp,
            "psnr": psnr,
            "ms-ssim": ms_ssim,
            "encoding_time": enc_time,
            "decoding_time": dec_time,
        }
    
    
    def get_encoder_path(self, build_dir):
        elfnames = {
            "Linux": "EncoderAppStatic",
            "Darwin": "EncoderApp",
        }
        if platform.system() not in elfnames:
            raise NotImplementedError(f"Unsupported platform: {platform.system()}")
        return os.path.join(build_dir, elfnames[platform.system()])

    
    def get_decoder_path(self, build_dir):
        elfnames = {
            "Linux": "DecoderAppStatic",
            "Darwin": "DecoderApp",
        }
        if platform.system() not in elfnames:
            raise NotImplementedError(f"Unsupported platform: {platform.system()}")
        return os.path.join(build_dir, elfnames[platform.system()])
