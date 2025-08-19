"""
When a new training task is added, use this script to automatically register the model classes defined in models.py files.

The file structure is expected to be like this:
├─ lossy_image_compression
│  ├─ charm
│  │  └─ models.py (** models needed to be registered **)
│  ├─ compressai_impl
│  │  └─ models.py
│  ├─ ...
│  ├─ module.py (** training and validating steps **)
└─ video_compression
   └─ ... (similar with lossy_image_compression)
"""

import os
import importlib
from compresslab.utils.registry import ModelRegistry
from compresslab.utils.constant import MODEL_DEFAULT_FILENAME
import inspect
from torch.nn import Module

SKIP_DIRS = ["__pycache__", "base", 
             "video_compression"]   # TODO: fix later

current_dir = os.path.dirname(os.path.abspath(__file__))

for task_dir in os.listdir(current_dir):
    if task_dir not in SKIP_DIRS:
        for root, dirs, files in os.walk(os.path.join(current_dir, task_dir)):
            for file in files:
                if file == MODEL_DEFAULT_FILENAME:
                    models_dir = os.path.join(root, file)

                    spec = importlib.util.spec_from_file_location(MODEL_DEFAULT_FILENAME, models_dir)
                    models_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(models_module)

                    classes = [
                        cls for name, cls in inspect.getmembers(models_module, inspect.isclass)
                        if issubclass(cls, Module) and cls.__module__ == models_module.__name__
                    ]

                    for cls in classes:
                        ModelRegistry.register(cls.__name__, define_path=models_dir)(cls)