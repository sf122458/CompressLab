from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
import yaml, json
from pydantic_yaml import parse_yaml_file_as
from pathlib import Path

class DataSetting(BaseModel):
    Key: str
    Params: Optional[Dict[str, Any]] = None

class ModelSetting(BaseModel):
    Name: str = None
    Key: str
    Params: Optional[Dict[str, Any]] = None
    ExtParams: Optional[Dict[str, Any]] = None

class BenchmarkItem(BaseModel):
    Key: str
    Params: Optional[Dict[str, Any]] = None

class TrainClass(BaseModel):
    Epoch: int
    Valinterval: int
    Output: str = "output"
    Benchmark: Optional[List[BenchmarkItem]] = None


class EnvClass(BaseModel):
    Devices: List[int] = [0]


class Config(BaseModel):
    Global: Optional[Dict[str, Any]] = None
    Model: List[ModelSetting]
    Data: DataSetting
    Train: TrainClass
    Env: EnvClass
