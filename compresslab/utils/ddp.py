import torch
from typing import Dict
import logging
import os, glob
from compresslab.config import Config
from compresslab.utils.registry import TrainerRegistry

def ddpTraining(
        config: Config
    ):

    os.makedirs(config.Train.output, exist_ok=True)
    os.system(f"cp {config.Parser.Config} {config.Train.Output}/config.yaml")

    for model_name, v in config.Model.Net.items():
        assert isinstance(v, Dict), "Model parameters should be a dictionary."
        trainer = TrainerRegistry.get(config.Train.Trainer if config.Train.Trainer is not None else "Default")(
            config=config,
            model_key=v['key'], 
            model_params=v['params'], 
            model_name=model_name)

        if not config.Parser.Testonly:
            trainer.train()
        trainer.test()

    logging.info("Finish training.")