from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import Dict, Any, Optional, List, Union
import yaml, os
from yaml.nodes import ScalarNode, SequenceNode, MappingNode
import re

class NumericStringConverterMixin(BaseModel):
    _convert_fields: List[str] = []

    @field_validator('*', mode='before')
    def convert_string_values_to_numeric(cls, v: Any, info: ValidationInfo) -> Any:
        if info.field_name not in cls._convert_fields.get_default():
            return v
        
        def _recursive_convert(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: _recursive_convert(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [_recursive_convert(item) for item in value]
            elif isinstance(value, str):
                try:
                    return int(value)
                except (ValueError, TypeError):
                    pass
                try:
                    return float(value)
                except (ValueError, TypeError):
                    pass
                return value
            else:
                return value

        return _recursive_convert(v)

########## Data Setting ##########
class DatasetConfig(NumericStringConverterMixin):
    _convert_fields = ['Params']
    
    Key: str = Field(description="Registered name of the dataset")
    Params: Optional[Dict[str, Any]] = Field(default_factory=dict,
                                             description="Parameters for the dataset, " \
                                             "provided as a dictionary.",
                                             kw_only=True
                                             )

class DataSetting(BaseModel):
    """Dataset setting for the datamodule."""
    Train: DatasetConfig = Field(default=None, description="Configuration for the training dataset.")
    Val: DatasetConfig = Field(default=None, description="Configuration for the validation dataset.")
    Test: Union[DatasetConfig, List[DatasetConfig]] = Field(description="Configuration for the testing dataset.")
    BatchSize: int = Field(default=32, description="Batch size for training.")
    NumWorkers: int = Field(default=4, description="Number of workers for data loading.")

########## Codec Classes ##########

class OptimizerConfig(NumericStringConverterMixin):
    _convert_fields = ['Params']
    
    Key: str = Field(default="Adam", description="Registered name of the optimizer.")
    Lr: float = Field(default=1e-4, description="Learning rate for the optimizer.")
    Params: Optional[Dict[str, Union[str, float, int, List, Dict]]] = Field(default_factory=dict,
                                             description="Parameters for the optimizer, "\
                                             "provided as a dictionary.",
                                             kw_only=True
                                             )

class GeneralCodecExtParams(BaseModel):
    model_config = {
        "extra": "forbid"
    }
    
    SaveRecon: bool = Field(
        default=False, description="Whether to save reconstructed images/videos.",
    )
    
    SaveBitstream: bool = Field(
        default=False, description="Whether to save the bitstream during testing. \
            The time cost of writing and reading the bitstream may also be contained in the final metrics."
    )
    
    Optimizer: OptimizerConfig = Field(
        default_factory=OptimizerConfig,
        description="Configuration for the optimizer."
    )

    GradAccumulateBatches: int = Field(
        default=1, description="Number of batches to accumulate gradients over before performing an optimizer step."
    )
    
    # TODO: a more elegant way?
    FinetuneRatio: Optional[float] = Field(
        default=1.0, description="The ratio of training steps for MS-SSIM model fine-tuning."
    )
    FinetuneStep: Optional[int] = Field(
        default=-1, description="The ratio of training steps for MS-SSIM model fine-tuning." \
        "If `FinetuneStep` <= 0, `FinetuneRatio` will be used else `FinetuneStep` will override the `FinetuneRatio`."
    )
    
    # Compression levels
    Lmbda: Union[float, List[float], Dict[str, List[float]]] = Field(
        default=0.0018,
        description="Compression levels for the codec." \
        "It supports a single float value, a list of floats, or a dictionary with keys 'mse' and 'ms_ssim'." \
        "If you set the fine-tuning step with `FinetuneRatio` or `FinetuneStep`, `Lmbda` must be a dictionary with keys 'mse' and 'ms_ssim'." \
    )
    
    # NOTE: `vmap` can't always accelerate the forward pass, use with caution.
    VmapForward: bool = Field(
        default=False, description="Whether to use vmap for forward pass."
    )

class GeneralCodec(NumericStringConverterMixin):
    """Base class for general codecs.
    """
    _convert_fields = ['Params']
    
    Name: Optional[str] = Field(default=None, description="Self-defined name for the codec")
    Key: str = Field(description="Registered name of the codec")
    Params: Optional[Dict[str, Any]] = Field(default_factory=dict, 
                                             description="Parameters for the codec, " \
                                             "provided as a dictionary.",
                                             kw_only=True
                                             )
    ExtParams: Optional[GeneralCodecExtParams] = Field(
        default_factory=GeneralCodecExtParams,
        description="Additional parameters for the codec.",
    )


########### Benchmark Item ##########
class BenchmarkItem(BaseModel):
    Key: str = Field(
        description="Registered name of the benchmark item."
    )
    Params: Optional[Dict[str, Any]] = Field(default_factory=dict, 
                                             description="Parameters for the benchmark item, " \
                                             "provided as a dictionary.",
                                             kw_only=True
                                             )

########### Training Configuration ##########
class TrainClass(BaseModel):
    Epoch: int = Field(default=None, description="Number of epochs to train the model.")
    Steps: int = Field(default=-1, description="Number of steps to train the model.")
    Valinterval: Optional[int] = Field(
        default=1, description="Validation interval in epochs."
    )
    Benchmark: Optional[List[BenchmarkItem]] = None

############ Environment Configuration ##########
class EnvClass(BaseModel):
    Devices: List[int] = [0]


class Config(BaseModel):
    Global: Optional[Dict[str, Any]] = None
    Model: List[GeneralCodec]
    Data: DataSetting
    Train: TrainClass = Field(default_factory=TrainClass)
    Env: EnvClass = Field(default_factory=EnvClass)



class Loader(yaml.SafeLoader):
    """Custom YAML loader.
    It supports the `!include` tag to include other YAML files.
    1. If the `!include` tag is used in a sequence, and the included file also contains a sequence,
       and its elements will be merged into the parent sequence, otherwise, the included file will be directly replaced.
    2. If the `!include` tag is used in a mapping, the included file can be of any type,
       and it will replace the value in the parent mapping.
       
    NOTE: The path of the included file is relative to the including file.
    """
    def __init__(self, stream):
        self._base_dir = os.path.dirname(os.path.abspath(stream.name))
        super().__init__(stream)
        
    def compose_sequence_node(self, anchor):
        node = super().compose_sequence_node(anchor)
        
        new_value = []
        for child in node.value:
            if isinstance(child, ScalarNode) and child.tag == '!include':
                include_node = self._process_include_node(child)
                if isinstance(include_node, SequenceNode):
                    new_value.extend(include_node.value)
                elif isinstance(include_node, MappingNode) and include_node.tag == '!loop':
                    new_value.extend(self._process_loop_node(include_node).value)
                else:
                    new_value.append(include_node)
            elif isinstance(child, MappingNode) and child.tag == '!loop':
                new_value.extend(self._process_loop_node(child).value)
            else:
                new_value.append(child)
        
        node.value = new_value
        return node

    def compose_mapping_node(self, anchor):
        node = super().compose_mapping_node(anchor)
        new_value = []
        
        for key_node, value_node in node.value:
            if isinstance(value_node, ScalarNode) and value_node.tag == '!include':
                include_node = self._process_include_node(value_node)
                new_value.append((key_node, include_node))
            elif isinstance(value_node, MappingNode) and value_node.tag == '!loop':
                loop_node = self._process_loop_node(value_node)
                new_value.append((key_node, loop_node))
            else:
                new_value.append((key_node, value_node))
        
        node.value = new_value
        return node
    
    def _process_include_node(self, include_node: ScalarNode) -> Union[ScalarNode, SequenceNode, MappingNode]:
        include_path = os.path.join(
            self._base_dir, 
            self.construct_scalar(include_node)
        )
        with open(include_path, 'r') as f:
            include_loader = Loader(f)
            include_node = include_loader.get_single_node()
        return include_node
    
    def _process_loop_node(self, loop_node: MappingNode) -> SequenceNode:
        loop_config = self.construct_mapping(loop_node, deep=True)
        self._validate_loop_config(loop_config)
        
        vars_dict = loop_config['vars']
        template = loop_config['template']
        var_names = list(vars_dict.keys())
        var_lists = list(vars_dict.values())
        total_items = len(var_lists[0])
        
        loop_items = []
        for idx in range(total_items):
            var_mapping = {var_names[i]: var_lists[i][idx] for i in range(len(var_names))}
            item_data = self._replace_placeholders(template, var_mapping)
            item_data = MappingNode(tag='tag:yaml.org,2002:map', value=[
                (ScalarNode(tag='tag:yaml.org,2002:str', value=k), 
                    ScalarNode(tag='tag:yaml.org,2002:str', value=v))
                for k, v in item_data.items()
            ])
            loop_items.append(item_data)
        
        return SequenceNode(tag='tag:yaml.org,2002:seq', value=loop_items)
    
    def _replace_placeholders(self, node, var_mapping):
        if isinstance(node, dict):
            return {
                k: self._replace_placeholders(v, var_mapping)
                for k, v in node.items()
            }
        elif isinstance(node, list):
            return [
                self._replace_placeholders(item, var_mapping)
                for item in node
            ]
        elif isinstance(node, str):
            result = re.sub(r'\$\{\s*(\w+)\s*\}', lambda m: str(var_mapping.get(m.group(1), m.group(0))), node)
            # Try to convert back to original data type
            if result != node:
                try:
                    return int(result)
                except ValueError:
                    pass
                try:
                    return float(result)
                except ValueError:
                    pass
                if result.lower() in ('true', 'false'):
                    return result.lower() == 'true'
                if result.lower() in ('null', 'none', '~'):
                    return None
            return result
        elif isinstance(node, ScalarNode):
            scalar_value = self.construct_scalar(node)
            return self._replace_placeholders(scalar_value, var_mapping)
        else:
            return node
        
    def _validate_loop_config(self, loop_config):
        if 'vars' not in loop_config or 'template' not in loop_config:
            raise ValueError("!loop tag must contain 'vars' and 'template' keys.")

        vars_dict = loop_config['vars']
        if not isinstance(vars_dict, dict):
            raise ValueError("'vars' must be a dictionary.")
        for var_name, var_list in vars_dict.items():
            if not isinstance(var_name, str) or not isinstance(var_list, list):
                raise ValueError(f"!loop") # TODO
            
        var_lists = list(vars_dict.values())
        if not var_lists:
            raise ValueError("!loop 'vars' cannot be empty.")
        list_lengths = [len(lst) for lst in var_lists]
        if len(set(list_lengths)) != 1:
            raise ValueError("All lists in !loop must have the same length.")
        if list_lengths[0] == 0:
            raise ValueError("Lists in !loop cannot be empty.")
    