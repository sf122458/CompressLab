from .models import DVC
from compresslab.utils.registry import ModelRegistry

ModelRegistry.register("DVC")(DVC)