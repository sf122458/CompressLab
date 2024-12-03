import yaml
import inspect
from pathlib import Path

from compresslab.config import Config
import compresslab.utils.log
import os
import glob
import logging
import compresslab.utils
from compresslab.utils.ddp import ddpTraining
from compresslab.utils.registry import *
import argparse

# register all the classes
import compresslab.nn
import compresslab.loss
import compresslab.optim
import compresslab.train
import compresslab.utils.registry

def main(args):
    if args.config is None:
            raise ValueError("Please provide a config file.")
    config = Config.deserialize(yaml.full_load(Path(args.config).read_text()))

    # If the output ckpt exist, resume training.
    config.Train.output = os.path.join(config.Train.Output, get_exp_name(config))
    os.makedirs(os.path.join(config.Train.Output, 'ckpt'), exist_ok=True)

    if not os.path.exists(os.path.join(config.Train.output, 'config.yaml')):
        os.system(f"cp {args.config} {config.Train.output}/config.yaml")
    
    ckpt_list = glob.glob(os.path.join(config.Train.Output, 'ckpt', '*.ckpt'))
    if len(ckpt_list) == 0:
        resume = None
        logging.info("Start training from the beginning.")
    else:
        resume = sorted(ckpt_list)[-1]
        logging.info(f"Restore from the checkpoint {resume}")

    ddpTraining(config, resume, args)

def get_exp_name(config: Config):
    loss_setting = "_".join([f"{k}_{v}" for k, v in config.Train.Loss.items()])
    return f"{config.Model.Net.Key}_{loss_setting}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--config', required=False, type=str,help ='Config file path.', default=None)
    parser.add_argument('--list', action="store_true", help='List all available models.')
    parser.add_argument('--test_only', action="store_true", help='Test only.')
    args = parser.parse_args()
    if args.list:
        registry_name = [clsname for (clsname, _) in inspect.getmembers(compresslab.utils.registry, inspect.isclass) 
                         if issubclass(getattr(compresslab.utils.registry, clsname), Registry)]
        for registry in registry_name:
            if registry == "Registry":
                continue
            print(registry)
            print(getattr(compresslab.utils.registry, registry).summary())
            print('-'*150)
    else:
        main(args=args)