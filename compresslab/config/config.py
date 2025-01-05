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


class Config(BaseModel):
    Model: ModelClass
    Train: TrainClass
    Env: EnvClass
    Parser: Optional[ParserClass] = ParserClass()


if __name__ == "__main__":
    info = yaml.full_load(Path("/home/gpu-4/lyx/compress_lab/config/template.yaml").read_text())
    print(info)
    info = json.dumps(info)
    config = Config.model_validate_json(info)
    config.Parser.Testonly = True