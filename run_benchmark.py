import yaml
from pathlib import Path
import logging
from cbench.config.config import Config
import os
from cbench.utils.ddp import ddpTraining
from cbench.utils.registry import ModelRegistry
import argparse

import cbench.nn.model
import cbench.loss

def main(configPath: str):
    configPath = Path(configPath)
    loggingLevel = logging.INFO
    logging.basicConfig(level=loggingLevel)

    config = Config.deserialize(yaml.full_load(configPath.read_text()))

    model = ModelRegistry.get("Balle")()

    # If the output ckpt exist, resume training.
    if os.path.exists(os.path.join(config.Train.trainSet, 'latest', 'saved.ckpt')):
        # resume = os.path.join(os.path.join(config.Train.SaveDir))
        # ckpt = torch.load(resume, "cpu")
        logging.info(f"Restore from the checkpoint {resume}")
        # resume = Path(resume)
    else:
        resume = None
        logging.info("Start training from the beginning.")

    
    ddpTraining(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--config', required=True, type=str,help ='Config file path.')
    args = parser.parse_args()
    main(configPath=args.config)