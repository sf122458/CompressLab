import torch
from typing import Dict
import wandb
import logging
from tensorboardX import SummaryWriter
import datetime
import os, glob
from compresslab.config import Config
from compresslab.utils.registry import TrainerRegistry

def ddpTraining(
        config: Config, 
        args
    ):

    os.system(f"cp {args.config} {config.Train.output}/config.yaml")

    for model_name, v in config.Model.Net.items():
        assert isinstance(v, Dict), "Model parameters should be a dictionary."
        # print(f"Training {name}...")
        trainer = TrainerRegistry.get(config.train.Trainer if config.train.Trainer is not None else "Default")(
            config=config, 
            args=args, 
            model_key=v.Key, 
            model_params=v.Params, 
            model_name=model_name)

        if not args.test_only:
            trainer.train()
        trainer.test()

    logging.info("Finish training.")