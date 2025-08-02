import torch
import os
import sys
import subprocess
from typing import Union, Dict
import numpy as np
from PIL import Image
from pytorch_msssim import ms_ssim


HM_BUILD_DIR = "third_party/HM/bin"
VTM_BUILD_DIR = "third_party/VTM/bin"

IMG_EXTENSIONS = {
    ".jpg",
    ".png"
}

YCBCR_WEIGHTS = {
    # Spec: (K_r, K_g, K_b) with K_g = 1 - K_r - K_b
    "ITU-R_BT.709": (0.2126, 0.7152, 0.0722)
}

def _check_input_tensor(tensor: torch.Tensor) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or not tensor.is_floating_point()
        or len(tensor.size()) not in (3, 4)
        or not tensor.size(-3) == 3
    ):
        raise ValueError(
            "Expected a 3D or 4D tensor with shape (Nx3xHxW) or (3xHxW) as input"
        )

def rgb2ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    """RGB to YCbCr conversion for torch Tensor.
    Using ITU-R BT.709 coefficients.

    Args:
        rgb (torch.Tensor): 3D or 4D floating point RGB tensor

    Returns:
        ycbcr (torch.Tensor): converted tensor
    """

    r, g, b = rgb.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5
    ycbcr = torch.cat((y, cb, cr), dim=-3)
    return ycbcr

def ycbcr2rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    """YCbCr to RGB conversion for torch Tensor.
    Using ITU-R BT.709 coefficients.

    Args:
        ycbcr (torch.Tensor): 3D or 4D floating point RGB tensor

    Returns:
        rgb (torch.Tensor): converted tensor
    """
    _check_input_tensor(ycbcr)

    y, cb, cr = ycbcr.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    rgb = torch.cat((r, g, b), dim=-3)
    return rgb


def get_filesize(filepath: str) -> int:
    """Return file size in bits of `filepath`."""
    if not os.path.isfile(filepath):
        raise ValueError(f'Invalid file "{filepath}".')
    return os.stat(filepath).st_size


def read_image(filepath: str, mode: str = "RGB") -> np.array:
    """Return PIL image in the specified `mode` format."""
    if not os.path.isfile(filepath):
        raise ValueError(f'Invalid file "{filepath}".')
    return Image.open(filepath).convert(mode)

def run_command(cmd, ignore_returncodes=None):
    cmd = [str(c) for c in cmd]
    try:
        rv = subprocess.check_output(cmd)
        return rv.decode("ascii")
    except subprocess.CalledProcessError as err:
        if ignore_returncodes is not None and err.returncode in ignore_returncodes:
            return err.output
        print(err.output.decode("utf-8"))
        sys.exit(1)


def get_ffmpeg_version():
    rv = run_command(["ffmpeg", "-version"])
    return rv.split()[2]


def get_bpg_version(encoder_path):
    rv = run_command([encoder_path, "-h"], ignore_returncodes=[1])
    return rv.split()[4]

def _compute_psnr(a, b, max_val: float = 255.0) -> float:
    mse = torch.mean((a - b) ** 2).item()
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr

def _compute_ms_ssim(a, b, max_val: float = 255.0) -> float:
    return ms_ssim(a, b, data_range=max_val).item()

def compute_metrics(
    a: Union[np.array, Image.Image],
    b: Union[np.array, Image.Image],
    max_val: float = 255.0,
) -> Dict[str, float]:
    """Returns PSNR and MS-SSIM between images `a` and `b`."""
    def _convert(x):
        if isinstance(x, Image.Image):
            x = np.asarray(x)
        x = torch.from_numpy(x.copy()).float().unsqueeze(0)
        if x.size(3) == 3:
            # (1, H, W, 3) -> (1, 3, H, W)
            x = x.permute(0, 3, 1, 2)
        return x

    a = _convert(a)
    b = _convert(b)

    return _compute_psnr(a, b, max_val), _compute_ms_ssim(a, b, max_val)
