import torch
from typing import Dict
import logging
import os, glob
from compresslab.config import Config
from compresslab.utils.registry import TrainerRegistry

def ddpTraining(
        config: Config
    ):

    os.makedirs(config.Train.Output, exist_ok=True)
    os.system(f"cp {config.Parser.Config} {config.Train.Output}/config.yaml")

    for model_name, v in config.Model.Net.items():
        trainer = TrainerRegistry.get(config.Train.Trainer if config.Train.Trainer is not None else "Default")(
            config=config,
            model_key=v.Key,
            model_params=v.Params, 
            model_name=model_name, 
            loss_config=v.Loss)

        if not config.Parser.Testonly:
            trainer.train()
        trainer.test()

    logging.info("Finish training.")