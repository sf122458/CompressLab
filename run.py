import yaml, json
import inspect
from pathlib import Path
import argparse
import logging
import os
from compresslab.utils.config import Config
from compresslab.utils.registry import Registry, DataRegistry, ModelRegistry
from lightning import Trainer
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from argparse import Namespace
import compresslab.nn
import compresslab.data
from pydantic_yaml import parse_yaml_file_as

class Args(Namespace):
    config: str = None
    list: bool = False
    test_only: bool = False

def main(args: Args):
    if args.list:
        import compresslab.utils.registry
        registry_name = [clsname for (clsname, _) in inspect.getmembers(compresslab.utils.registry, inspect.isclass) 
                         if issubclass(getattr(compresslab.utils.registry, clsname), Registry) if clsname != "Registry"]
        for registry in registry_name:
            getattr(compresslab.utils.registry, registry).summary()
            
    else:
        if args.config is None:
            raise ValueError("Please provide a config file.")
        
        config = parse_yaml_file_as(Config, args.config)

        datamodule = DataRegistry.get(config.Data.Key)(**config.Data.Params)


        for model in config.Model:
            modelmodule = ModelRegistry.get(model.Key)(**model.Params)

            out_dir = os.path.join(config.Train.Output, model.Key)

            trainer = Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=config.Env.Devices,
                strategy="ddp",
                max_epochs=config.Train.Epoch,
                check_val_every_n_epoch=config.Train.Valinterval,
                default_root_dir=out_dir,
                callbacks=[
                    RichProgressBar(),
                    ModelCheckpoint(
                        dirpath=os.path.join(out_dir, "checkpoints"),
                        every_n_epochs=config.Train.Valinterval,
                        save_last=True,
                    )
                ],
                logger=True,
            )
            if not args.test_only:
                trainer.fit(modelmodule, datamodule, ckpt_path="last")
            
            # trainer.test(modelmodule, datamodule, ckpt_path="last")

        logging.info("Finish training.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--config', required=False, type=str,help ='Config file path.', default=None)
    parser.add_argument('--list', action="store_true", help='List all available models.')
    parser.add_argument('--test_only', action="store_true", help='Test only.')
    
    args = parser.parse_args()
    main(args)