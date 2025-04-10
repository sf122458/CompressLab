from .dataset import *
from compresslab.utils.registry import DataRegistry

for module in [
    BasicImageDataModule
]:
    DataRegistry.register(module.__name__)(module)