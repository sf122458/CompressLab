import yaml
import inspect
from pathlib import Path
import logging
from compresslab.config.config import Config
import os
import compresslab.utils
from compresslab.utils.ddp import ddpTraining
from compresslab.utils.registry import *
import argparse

# register all the classes
import compresslab.nn
import compresslab.loss
import compresslab.optim
import compresslab.train.trainer
import compresslab.utils.registry

def main(configPath: Path):

    # Logging level setting.
    loggingLevel = logging.DEBUG
    logging.basicConfig(level=loggingLevel)

    config = Config.deserialize(yaml.full_load(configPath.read_text()))

    # If the output ckpt exist, resume training.
    if os.path.exists(os.path.join(config.Train.TrainSet.Path, 'latest', 'saved.ckpt')):
        resume = os.path.join(os.path.join(config.Train.SaveDir))
        # ckpt = torch.load(resume, "cpu")
        logging.info(f"Restore from the checkpoint {resume}")
        # resume = Path(resume)
    else:
        resume = None
        logging.info("Start training from the beginning.")

    
    ddpTraining(config, resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--config', required=False, type=str,help ='Config file path.', default=None)
    parser.add_argument('--list', action="store_true", help='List all available models.')
    args = parser.parse_args()
    if args.list:
        registry_name = [clsname for (clsname, _) in inspect.getmembers(compresslab.utils.registry, inspect.isclass)]
        for registry in registry_name:
            if registry == "Registry":
                continue
            print(registry)
            print(getattr(compresslab.utils.registry, registry).summary())

    else:
        if args.config is None:
            raise ValueError("Please provide a config file.")
        main(configPath=Path(args.config))