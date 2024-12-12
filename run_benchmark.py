import yaml, json
import inspect
from pathlib import Path

# from compresslab.config import Config
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
    # config = Config.deserialize(yaml.full_load(Path(args.config).read_text()))
    # config.Train.output = os.path.join('output', config.Train.Output)

    config = Config.model_validate_json(json.dumps(yaml.full_load(Path("/home/gpu-4/lyx/compress_lab/config/t2.yaml").read_text())))
    config.Parser.Config = args.config
    if args.test_only:
        config.Parser.Testonly = True

    ddpTraining(config)


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