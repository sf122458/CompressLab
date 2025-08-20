from .base import DataModule

# register dataset class automatically
import os
import importlib
from compresslab.utils.registry import DataRegistry
import inspect
from torch.utils.data import Dataset

current_dir = os.path.dirname(os.path.abspath(__file__))

SKIP_FILES = ["__init__.py", "base.py",
              "video.py", "rawvideo.py"]   # TODO: fix later


for file in os.listdir(current_dir):
    file_path = os.path.join(current_dir, file)
    if os.path.isfile(file_path) and file not in SKIP_FILES:
        # Convert path to module format and find compresslab
        module_path = file_path.replace(os.sep, '.').replace('.py', '')
        compresslab_index = module_path.rfind('compresslab')
        if compresslab_index != -1:
            module_path = module_path[compresslab_index:]

        models_module = importlib.import_module(module_path)

        classes = [
            cls for name, cls in inspect.getmembers(models_module, inspect.isclass)
            if issubclass(cls, Dataset) and cls.__module__ == models_module.__name__
        ]

        for cls in classes:
            DataRegistry.register(cls)