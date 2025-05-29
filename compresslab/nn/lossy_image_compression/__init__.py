"""
When a new training task is added, use this script to automatically register the model classes defined in models.py files.

The file structure is expected to be like this:
├─ lossy_image_compression
│  ├─ charm
│  │  └─ models.py (** the definition of end-to-end models needed to be registered **)
│  ├─ compressai_impl
│  │  └─ models.py
│  ├─ ...
│  ├─ module.py (** responsible for defining the LightningModule for training **)
└─ video_compression
   └─ ... (similar with lossy_image_compression)
"""

import os
import importlib
from compresslab.utils.registry import ModelRegistry
import inspect
from torch.nn import Module

current_dir = os.path.dirname(os.path.abspath(__file__))
for root, dirs, files in os.walk(current_dir):
    for file in files:
        if file == "models.py":
            models_dir = os.path.join(root, file)

            spec = importlib.util.spec_from_file_location("models", models_dir)
            models_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(models_module)

            classes = [
                cls for name, cls in inspect.getmembers(models_module, inspect.isclass)
                if issubclass(cls, Module) and cls.__module__ == models_module.__name__
            ]

            for cls in classes:
                ModelRegistry.register(cls.__name__, define_path=models_dir)(cls)