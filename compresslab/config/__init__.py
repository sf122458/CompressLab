"""
Class decorated with `dataclass` corresponds with the configuration in yaml files.
The `Schema` classes are used to serialize and deserialize the dataclass objects, and can also validate and raise error if some required keys miss.
"""

# from dataclasses import dataclass
# from marshmallow import Schema, fields, post_load, RAISE
# from typing import Dict, List, Any

# @dataclass
# class General:
#     key: str
#     params: Dict[str, Any]

#     @property
#     def Key(self) -> str:
#         return self.key
    
#     @property
#     def Params(self) -> Dict[str, Any]:
#         return self.params
    
# @dataclass
# class Model:
#     compound: str
#     # net: General
#     net: Dict[str, General]

#     @property
#     def Compound(self) -> str:
#         return self.compound
    
#     @property
#     def Net(self) -> Dict[str, General]:
#         return self.net


# @dataclass
# class Dataset:
#     path: str
#     transform: Dict[str, Any]=None

#     @property
#     def Path(self) -> str:
#         return self.path
    
#     @property
#     def Transform(self) -> Dict[str, Any]:
#         return self.transform


    
# @dataclass
# class Log:
#     wandb_enable: bool
#     wandb_project: str=None
#     wandb_name: str=None
    
# @dataclass
# class ENV:
#     WANDB_API_KEY: str=None

#     CUDA_VISIBLE_DEVICES: str=None
#     NUM_WORKERS: int=None

# class GeneralSchema(Schema):
#     class Meta:
#         unknown = RAISE
#     key = fields.Str(required=True, description="")
#     params = fields.Dict(required=True, description="")

#     @post_load
#     def _(self, data, **kwargs):
#         return General(**data)
    
# class ModelSchema(Schema):
#     class Meta:
#         unknown = RAISE
#     compound = fields.Str(required=True, description="")
#     # net = fields.Nested(GeneralSchema(), required=True)
#     net = fields.Dict(required=True, description="")

#     @post_load
#     def _(self, data, **kwargs):
#         return Model(**data)
    
# class DatasetSchema(Schema):
#     class Meta:
#         unknown = RAISE
#     path = fields.Str(required=True, description="Path to the dataset.")
#     transform = fields.Dict(required=False, description="Transforms to apply to the dataset.")

#     @post_load
#     def _(self, data, **kwargs):
#         return Dataset(**data)

# @dataclass
# class Train:
#     batchsize: int
#     epoch: int
#     valinterval: int
#     trainset: Dataset
#     valset: Dataset
#     output: str
#     loss: Dict[str, Any]
#     optim: General
#     schdr: General=None
#     trainer: str=None

#     @property
#     def BatchSize(self) -> int:
#         return self.batchsize

#     @property
#     def Epoch(self) -> int:
#         return self.epoch
    
#     @property
#     def ValInterval(self) -> int:
#         return self.valinterval

#     @property
#     def TrainSet(self) -> Dataset:
#         return self.trainset
    
#     @property
#     def ValSet(self) -> Dataset:
#         return self.valset
    
#     @property
#     def Output(self) -> str:
#         return self.output
    
#     @property
#     def Loss(self) -> Dict[str, Any]:
#         return self.loss
    
#     @property
#     def Optim(self) -> General:
#         return self.optim
    
#     @property
#     def Schdr(self) -> General:
#         return self.schdr
    
#     @property
#     def Trainer(self) -> str:
#         return self.trainer

# class TrainSchema(Schema):
#     class Meta:
#         unknown = RAISE
#     batchsize = fields.Int(required=True, description="Batch size")
#     epoch = fields.Int(required=True, description="Epoch")
#     valinterval = fields.Int(required=True, description="Validation interval")
#     trainset = fields.Nested(DatasetSchema(), required=True)
#     valset = fields.Nested(DatasetSchema(), required=True)
#     output = fields.Str(required=True)
#     loss = fields.Dict(required=True)
#     optim = fields.Nested(GeneralSchema(), required=True)
#     schdr = fields.Nested(GeneralSchema(), required=False)
#     trainer = fields.Str(required=False)

#     @post_load
#     def _(self, data, **kwargs):
#         return Train(**data)
    
# class ENVSchema(Schema):
#     class Meta:
#         unknown = RAISE
#     # WANDB_ENABLE = fields.Bool(required=True)
#     WANDB_API_KEY = fields.Str(required=False)
#     # WANDB_PROJECT = fields.Str(required=False)
#     CUDA_VISIBLE_DEVICES = fields.Str(required=False)
#     NUM_WORKERS = fields.Int(required=False)

#     @post_load
#     def _(self, data, **kwargs):
#         return ENV(**data)
    
# class ConfigSchema(Schema):
#     class Meta:
#         unknown = RAISE
#     model = fields.Nested(ModelSchema(), required=True, description="Compress Model")
#     train = fields.Nested(TrainSchema(), required=True, description="Training configs.")
#     log = fields.Nested(GeneralSchema(), required=True, description="Logging configs.")
#     env = fields.Nested(ENVSchema(), required=False, description="Environment variables.")

#     @post_load
#     def _(self, data, **kwargs):
#         return Config(**data)

# @dataclass
# class Config:
#     model: Model
#     train: Train
#     log: General
#     env: ENV=None

#     @property
#     def Model(self) -> Model:
#         return self.model
    
#     @property
#     def Train(self) -> Train:
#         return self.train
    
#     @property
#     def Log(self) -> General:
#         return self.log

#     @property
#     def ENV(self) -> ENV:
#         return self.env
    
#     def serialize(self) -> Dict[str, Any]:
#         return ConfigSchema().dump(self)
    
#     @staticmethod
#     def deserialize(data: dict) -> "Config":
#         data = {key: value for key, value in data.items() if "$" not in key}
#         return ConfigSchema().load(data)

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import yaml, json
from pathlib import Path

class General(BaseModel):
    Key: str
    Params: Dict[str, Any]

class ModelSetting(BaseModel):
    Key: str
    Params: Dict[str, Any]
    Loss: Dict[str, Any]

class ModelClass(BaseModel):
    Compound: str
    Net: Dict[str, ModelSetting]

class TrainClass(BaseModel):
    Trainer: str
    Epoch: int
    Batchsize: int
    Valinterval: int
    Output: str
    Trainset: General
    Valset: General
    Optim: General
    Schdr: General
    Log: Optional[General] = None

class ParserClass(BaseModel):
    Testonly: bool = False
    Config: str = None

class EnvClass(BaseModel):
    WANDB_API_KEY: Optional[str] = None
    CUDA_VISIBLE_DEVICES: Optional[str] = None

class Config(BaseModel):
    Model: ModelClass
    Train: TrainClass
    Env: EnvClass
    Parser: Optional[ParserClass] = ParserClass()

if __name__ == "__main__":
    info = yaml.full_load(Path("/home/gpu-4/lyx/compress_lab/config/t2.yaml").read_text())
    print(info['Train'])
    info = json.dumps(info)
    config = Config.model_validate_json(info)
    config.Parser.Testonly = True