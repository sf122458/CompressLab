from .codec import *
from typing import Dict, Type

TRADITIONAL_CODEC: Dict[str, Type[Codec]] = {
    "JPEG": JPEG,
    "WebP": WebP,
    "JPEG2000": JPEG2000,
    "HM": HM,
    "VTM": VTM,
}