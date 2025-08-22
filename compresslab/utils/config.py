from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union

########## Data Setting ##########
class DatasetConfig(BaseModel):
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
    Test: DatasetConfig = Field(description="Configuration for the testing dataset.")
    BatchSize: int = Field(default=32, description="Batch size for training.")
    NumWorkers: int = Field(default=4, description="Number of workers for data loading.")

########## Codec Classes ##########
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
    
    # Extra parameters for the trainable codec.
    Lr: Optional[float] = Field(default=1e-4, description="Learning rate.")
    Auxlr: Optional[float] = Field(default=1e-3, description="Auxiliary learning rate for the entropy models.")
    
    
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
        "It supports a single float value, a list of floats, or a dictionary with keys 'mse' and 'ms-ssim'." \
        "If you set the fine-tuning step with `FinetuneRatio` or `FinetuneStep`, `Lmbda` must be a dictionary with keys 'mse' and 'ms-ssim'." \
    )
    
    # NOTE: `vmap` can't always accelerate the forward pass, use with caution.
    VmapForward: bool = Field(
        default=False, description="Whether to use vmap for forward pass."
    )

class GeneralCodec(BaseModel):
    """Base class for general codecs.
    """
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
