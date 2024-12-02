"""
    With the help of registry, we can dynamically load the instance from the configuration in .yaml files.
"""

from typing import Type, Generic, TypeVar, Dict, Union, List
import torch
import logging
import functools
import yaml
from io import StringIO
import re
import os
import inspect
from compresslab.utils.base import _baseTrainer, _baseCompound


# from the implementation in vlutils
def _alignYAML(str, pad=0, aligned_colons=False):
    props = re.findall(r'^\s*[\S]+:', str, re.MULTILINE)
    if not props:
        return str
    longest = max([len(i) for i in props]) + pad
    if aligned_colons:
        return ''.join([i+'\n' for i in map(
                    lambda str: re.sub(r'^(\s*.+?[^:#]): \s*(.*)',
                        lambda m: m.group(1) + ''.ljust(longest-len(m.group(1))-1-pad) + ':'.ljust(pad+1) + m.group(2), str, re.MULTILINE),
                    str.split('\n'))])
    else:
        return ''.join([i+'\n' for i in map(
                    lambda str: re.sub(r'^(\s*.+?[^:#]: )\s*(.*)',
                        lambda m: m.group(1) + ''.ljust(longest-len(m.group(1))+1) + m.group(2), str, re.MULTILINE),
                    str.split('\n'))])

def pPrint(d: dict) -> str:
    """Print dict prettier.

    Args:
        d (dict): The input dict.

    Returns:
        str: Resulting string.
    """
    with StringIO() as stream:
        yaml.safe_dump(d, stream, default_flow_style=False)
        return _alignYAML(stream.getvalue(), pad=1, aligned_colons=True)

T = TypeVar("T")

class Registry(Generic[T]):
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
    """
    _map: Dict[str, T]
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._map: Dict[str, T] = dict()

    @classmethod
    def register(cls, key):
        """Decorator for register anything into registry.

        Args:
            key (str): The key for registering an object.
        """
        if isinstance(key, str):
            def insert(value):
                cls._map[key] = {"cls": value, "path": inspect.stack()[1].filename}
            return insert
        else:
            cls._map[key.__name__] = key
            return key

    @classmethod
    def get(cls, key: str, default = None, logger: logging.Logger = logging.root) -> T:
        """Get an object from registry.

        Args:
            key (str): The key for the registered object.
        """
        result = cls._map.get(key, default)['cls']
        if result is None:
            logger.debug("Get None from \"%s\".", cls.__name__)
        elif isinstance(result, functools.partial):
            logger.debug("Get <%s.%s> from \"%s\".", result.func.__module__, result.func.__qualname__, cls.__name__)
        else:
            logger.debug("Get <%s.%s> from \"%s\".", result.__module__, result.__qualname__, cls.__name__)
        return result

    @classmethod
    def summary(cls) -> str:
        """Get registry summary.
        """
        return pPrint({
            k: v['cls'].__module__ + '.' + v['cls'].__name__ + ' registered in ' + v['path'] for k, v in cls._map.items()
        })

"""
Modules need to be registered.
Example:
    ```python
        ModelRegistry.register("class_key")(class_name)
    ```
        or
    ```python
        @ModelRegistry.register("class_key")
        class class_name:
            ...
    ```
"""
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