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

class ModelRegistry(Registry[Type["torch.nn.Module"]]):
    pass

class OptimizerRegistry(Registry[Type["torch.optim.Optimizer"]]):
    pass

class SchedulerRegistry(Registry[Type["torch.optim.lr_scheduler._LRScheduler"]]):
    pass

class LossRegistry(Registry[Type["torch.nn.Module"]]):
    pass
