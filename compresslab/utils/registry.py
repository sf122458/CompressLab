"""A registry. Inherit from it to create a lots of factories.

    Example:
    ```python
        # Inherit to make a factory.
        class Geometry(Registry):
            ...

        # Register with auto-key "Foo"
        @Geometry.register
        class Foo:
            ...

        # Register with manual-key "Bar"
        @Geometry.register("Bar")
        class Bar:
            ...

        instance = Geometry.get("Foo")()
        assert isinstance(instance, Foo)

        instance = Geometry["Bar"]()
        assert isinstance(instance, Bar)
    ```

    With the help of registry, we can dynamically load the instance according to the configuration in yaml files.
"""

from typing import Type
from vlutils.base.registry import Registry
import torch
import torchvision
from compresslab.utils.base import _baseTrainer, _baseCompound

class ModelRegistry(Registry[Type["torch.nn.Module"]]):
    pass

class CompoundRegistry(Registry[Type["_baseCompound"]]):
    pass

class OptimizerRegistry(Registry[Type["torch.optim.Optimizer"]]):
    pass

class SchedulerRegistry(Registry[Type["torch.optim.lr_scheduler._LRScheduler"]]):
    pass

class LossRegistry(Registry[Type["torch.nn.Module"]]):
    pass

class TrainerRegistry(Registry[Type["_baseTrainer"]]):
    pass

class TransformRegistry(Registry):
    pass
# class LayerRegistry(Registry[Type["torch.nn.Module"]]):
#     pass