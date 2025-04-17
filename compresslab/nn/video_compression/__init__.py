from .models import ScaleSpaceFlow
from compresslab.utils.registry import ModelRegistry

for model in [
    ScaleSpaceFlow
]:
    ModelRegistry.register(model.__name__)(model)