from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
import yaml, json
from pydantic_yaml import parse_yaml_file_as
from pathlib import Path

class General(BaseModel):
    Key: str
    Params: Optional[Dict[str, Any]] = None

class ModelSetting(BaseModel):
    Key: str
    Params: Optional[Dict[str, Any]]
    Lr: float = 1e-4
    Lmbda: Union[float, List[float]]

class TrainClass(BaseModel):
    Epoch: int
    Valinterval: int
    Output: str = "output"
    Benchmark: Optional[List[General]] = None


class EnvClass(BaseModel):
    Devices: List[int] = [0]


class Config(BaseModel):
    Global: Optional[Dict[str, Any]] = None
    Model: List[ModelSetting]
    Data: General
    Train: TrainClass
    Env: EnvClass
