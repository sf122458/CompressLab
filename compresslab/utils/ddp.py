import torch
from typing import Dict
import wandb
import logging
from tensorboardX import SummaryWriter
import datetime

from compresslab.config.config import Config
# from cbench.train.trainer import *
from compresslab.utils.registry import TrainerRegistry

def ddpTraining(
        config: Config, 
        resume  # TODO: checkpoint restore
    ):
    pass

    # should register all modules here?

    # TODO: DDP Settings
    if resume is not None:
        raise NotImplementedError

    # WANDB or Tensorboard
    run = None
    if config.Log.Key.upper() == "WANDB":
        logging.info("Use WANDB.")
        wandb.login(config.env.WANDB_API_KEY)
        config.Log.Params["name"] = config.Log.Params["name"] + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        run = wandb.init(
            # project=config.env.wandb_name,
            # config=None, # TODO
            # name=config.env.WANDB_NAME
            **config.Log.Params
        )
    elif config.Log.Key.upper() == "TENSORBOARD":
        logging.info("Use Tensorboard.")
        # TODO: Tensorboard
        run = SummaryWriter()
        raise NotImplementedError
    else:
        logging.warning("No logging service is enabled.")
    
    # trainer = ImageCompressionTrainer(config, 
    #                   log=run)
    logging.debug(config.Train.Trainer)
    trainer = TrainerRegistry.get(config.train.Trainer if config.train.Trainer is not None else "Default")(config=config, run=run)

    trainer.train()


    logging.info("Finish training.")