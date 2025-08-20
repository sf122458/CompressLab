from typing import Type, Generic, TypeVar, Dict
import torch
import logging
import functools
import yaml
from io import StringIO
import re
import inspect
from rich.console import Console
from rich.table import Table
from torch.utils.data import Dataset


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
    def register(cls, class_type, key = None):
        """Decorator for register anything into registry.

        Args:
            key (str): The key for registering an object.
        """
        assert isinstance(class_type, type), "Registered object must be a class."
        cls._map[key or class_type.__name__] = {"cls": class_type, "define_path": inspect.getfile(class_type)}
        # def insert(value):
        #     assert isinstance(value, type), "Registered object must be a class."
        #     cls._map[key or value.__name__] = {"cls": value, "define_path": define_path or inspect.getfile(value)}
        # return insert
    

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
        table = Table(title=cls.__name__)

        table.add_column("Name", justify="left", style="cyan", no_wrap=True)
        table.add_column("Path", justify="left", style="green")

        for k, v in cls._map.items():
            table.add_row(
                k,
                v['define_path'],
            )

        console = Console()
        console.print(table)

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

class DataRegistry(Registry[Type[Dataset]]):
    pass