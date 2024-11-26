"""
Class decorated with `dataclass` corresponds with the configuration in yaml files.
The `Schema` classes are used to serialize and deserialize the dataclass objects, and can also validate and raise error if some required keys miss.
"""

from dataclasses import dataclass
from marshmallow import Schema, fields, post_load, RAISE
from typing import Dict, List, Any

@dataclass
class General:
    key: str
    params: Dict[str, Any]

    @property
    def Key(self) -> str:
        return self.key
    
    @property
    def Params(self) -> Dict[str, Any]:
        return self.params

    
@dataclass
class GPU:
    gpus: int
    vRam: int
    wantsMore: bool

    @property
    def GPUs(self) -> int:
        return self.gpus

    @property
    def VRam(self) -> int:
        return self.vRam

    @property
    def WantsMore(self) -> bool:
        return self.wantsMore

@dataclass
class Train:
    batchSize: int
    epoch: int
    trainSet: str
    valSet: str
    output: str
    loss: Dict[str, Any]
    optim: General
    gpu: str
    schdr: General=None

    @property
    def BatchSize(self) -> int:
        return self.batchSize

    @property
    def Epoch(self) -> int:
        return self.epoch
    
    @property
    def TrainSet(self) -> str:
        return self.trainSet
    
    @property
    def ValSet(self) -> str:
        return self.valSet
    
    @property
    def Output(self) -> str:
        return self.output
    
    @property
    def Loss(self) -> Dict[str, Any]:
        return self.loss
    
    @property
    def Optim(self) -> General:
        # TODO
        raise NotImplementedError
    
    @property
    def Schdr(self) -> General:
        return self.schdr
    
    @property
    def GPU(self):
        return self.gpu
    
@dataclass
class ENV:
    WANDB_ENABLE: bool
    WANDB_API_KEY: str=None
    WANDB_PROJECT: str=None

    CUDA_VISIBLE_DEVICES: str=None

class GeneralSchema(Schema):
    class Meta:
        unknown = RAISE
    key = fields.Str(required=True, description="")
    params = fields.Dict(required=True, description="")

    @post_load
    def _(self, data, **kwargs):
        return General(**data)
    
class GPUSchema(Schema):
    class Meta:
        unknown = RAISE
    gpus = fields.Int(required=True, description="Number of gpus for training. This affects the `world size` of PyTorch DDP.", exclusiveMinimum=0)
    vRam = fields.Int(required=True, description="Minimum VRam required for each gpu. Set it to `-1` to use all gpus.")
    wantsMore = fields.Bool(required=True, description="Set to `true` to use all visible gpus and all VRams and ignore `gpus` and `vRam`.")

    @post_load
    def _(self, data, **kwargs):
        return GPU(**data)
    
class TrainSchema(Schema):
    class Meta:
        unknown = RAISE
    batchSize = fields.Int(required=True, description="Batch size")
    epoch = fields.Int(required=True, description="Epoch")
    trainSet = fields.Str(required=True)
    valSet = fields.Str(required=True)
    output = fields.Str(required=True)
    loss = fields.Dict(required=True)
    optim = fields.Nested(GeneralSchema(), required=True)
    schdr = fields.Nested(GeneralSchema(), required=False)
    gpu = fields.Nested(GPUSchema(), required=True)

    @post_load
    def _(self, data, **kwargs):
        return Train(**data)
    
class ENVSchema(Schema):
    class Meta:
        unknown = RAISE
    WANDB_ENABLE = fields.Bool(required=True)
    WANDB_API_KEY = fields.Str(required=False)
    WANDB_PROJECT = fields.Str(required=False)
    CUDA_VISIBLE_DEVICES = fields.Str(required=False)

    @post_load
    def _(self, data, **kwargs):
        return ENV(**data)
    
class ConfigSchema(Schema):
    class Meta:
        unknown = RAISE
    model = fields.Nested(GeneralSchema(), required=True, description="Compress Model")
    train = fields.Nested(TrainSchema(), required=True, description="Training configs.")
    env = fields.Nested(ENVSchema(), required=False, description="Environment variables.")

    @post_load
    def _(self, data, **kwargs):
        return Config(**data)

@dataclass
class Config:
    model: General
    train: Train
    env: ENV=None

    @property
    def Model(self):
        return self.model
    
    @property
    def Train(self) -> Train:
        return self.train
    
    @property
    def ENV(self) -> ENV:
        return self.env
    
    def serialize(self):
        return ConfigSchema().dump(self)
    
    @staticmethod
    def deserialize(data: dict) -> "Config":
        date = {key: value for key, value in data.items() if "$" not in key}
        return ConfigSchema().load(data)