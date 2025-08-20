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
                    
                    # Convert path to module format and find compresslab
                    module_path = models_dir.replace(os.sep, '.').replace('.py', '')
                    compresslab_index = module_path.rfind('compresslab')
                    if compresslab_index != -1:
                        module_path = module_path[compresslab_index:]

                    models_module = importlib.import_module(module_path)

                    classes = [
                        cls for name, cls in inspect.getmembers(models_module, inspect.isclass)
                        if issubclass(cls, Module) and cls.__module__ == models_module.__name__
                    ]

                    for cls in classes:
                        ModelRegistry.register(cls)