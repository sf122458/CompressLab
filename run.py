import inspect
from pathlib import Path
import argparse
import logging
import os
import torch
from compresslab.utils.config import Config
from compresslab.utils.registry import Registry, DataRegistry, ModelRegistry
from compresslab.utils.benchmark import Benchmark
import lightning as L
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary
from argparse import Namespace
import compresslab.nn
import compresslab.data
from pydantic_yaml import parse_yaml_file_as
import compresslab.utils.registry
import importlib.util


class Args(Namespace):
    config: str = None
    list: bool = False
    test_only: bool = False

def main(args: Args):
    if args.list:
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
            compressmodel = ModelRegistry.get(model.Key)(**model.Params)

            model_path = Path(getattr(compresslab.utils.registry, "ModelRegistry")._map.get(model.Key)["path"])

            module_file = model_path.parent / "module.py"
            if not module_file.exists():
                raise FileNotFoundError(f"module.py not found in {model_path.parent}")

            module_spec = importlib.util.spec_from_file_location("module", module_file)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)

            lightning_classes = [
                cls for name, cls in inspect.getmembers(module, inspect.isclass)
                if issubclass(cls, L.LightningModule)
            ]

            if not lightning_classes:
                raise ValueError(f"No class found in {module_file}")

            assert len(lightning_classes) == 1, f"Multiple LightningModule classes found in {module_file}"
            LightningModule = lightning_classes[0]  # Assuming the first match is the desired class
            
            modelmodule = LightningModule(compressmodel, lmbda=model.Lmbda, lr=model.Lr)

            exp_dir = os.path.join(config.Train.Output, Path(args.config).stem)
            os.makedirs(exp_dir, exist_ok=True)

            os.system(f"cp {args.config} {exp_dir}/config.yaml")

            out_dir = os.path.join(exp_dir, model.Key)

            trainer = Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=config.Env.Devices,
                strategy="ddp_find_unused_parameters_true",
                max_epochs=config.Train.Epoch,
                check_val_every_n_epoch=config.Train.Valinterval,
                default_root_dir=out_dir,
                callbacks=[
                    RichProgressBar(),
                    ModelCheckpoint(
                        dirpath=os.path.join(out_dir, "checkpoints"),
                        every_n_epochs=config.Train.Valinterval,
                        save_last=True,
                    ),
                    RichModelSummary(
                        max_depth=2,
                    )
                ],
                logger=TensorBoardLogger(save_dir=out_dir),
                deterministic=True # NOTE: this is important for reproducibility, otherwise the entropy decoding may fail
            )
            if not args.test_only:
                trainer.fit(modelmodule, datamodule, ckpt_path="last")
            
            trainer.test(modelmodule, datamodule, ckpt_path="last")

        
        Benchmark(exp_dir, config.Train.Benchmark)
        
        logging.info("Finish training.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--config', required=False, type=str,help ='Config file path.', default=None)
    parser.add_argument('--list', action="store_true", help='List all available models.')
    parser.add_argument('--test_only', action="store_true", help='Test only.')
    
    args = parser.parse_args()
    main(args)