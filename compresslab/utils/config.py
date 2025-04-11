from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
import yaml, json
from pydantic_yaml import parse_yaml_file_as
from pathlib import Path

class General(BaseModel):
    Key: str
    Params: Dict[str, Any]

class ModelSetting(BaseModel):
    Key: str
    Params: Dict[str, Any]
    Lmbda: Union[float, List[float]]
    Lr: float = 1e-4

class TrainClass(BaseModel):
    Epoch: int
    Valinterval: int
    Output: str = "output"
    Logger: Optional[General] = None


class EnvClass(BaseModel):
    Devices: List[int] = [0]


class Config(BaseModel):
    Global: Optional[Dict[str, Any]] = None
    Model: List[ModelSetting]
    Data: General
    Train: TrainClass
    Env: EnvClass


if __name__ == "__main__":
    config = parse_yaml_file_as(Config, "/home/gpu-4/lyx/Lightning/config/template.yaml")
    print(config)