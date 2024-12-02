import torch
from typing import Dict
import wandb
import logging
from tensorboardX import SummaryWriter
import datetime

from compresslab.config import Config
from compresslab.utils.registry import TrainerRegistry

def ddpTraining(
        config: Config, 
        resume,  # TODO: checkpoint restore
    ):

    # TODO: DDP Settings
    if resume is not None:
        # raise NotImplementedError
        logging.warning("DDP load is not implemented!")

    # WANDB or Tensorboard
    run = None
    if config.Log.Key.upper() == "WANDB":
        logging.info("Use WANDB.")
        wandb.login(config.env.WANDB_API_KEY)
        config.Log.Params["name"] = config.Log.Params["name"] + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        run = wandb.init(
            config={k:v for k, v in config.serialize().items() if k == 'model' or k == 'train'},
            **config.Log.Params
        )
    elif config.Log.Key.upper() == "TENSORBOARD":
        logging.info("Use Tensorboard.")
        # TODO: Tensorboard
        run = SummaryWriter(config.Log.Params)
        raise NotImplementedError
    else:
        logging.warning("No logging service is enabled.")
    
    
    trainer = TrainerRegistry.get(config.train.Trainer if config.train.Trainer is not None else "Default")(config=config, run=run, resume=resume)

    trainer.train()


    logging.info("Finish training.")