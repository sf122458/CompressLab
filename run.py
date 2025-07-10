import inspect
from pathlib import Path
import argparse
import logging
import os
import torch
import math
from compresslab.utils.config import Config
from compresslab.nn.base import BaseTrainer
from compresslab.utils.registry import Registry, DataRegistry, ModelRegistry
from compresslab.utils.benchmark import Benchmark
from compresslab.codec import TRADITIONAL_CODEC
import lightning as L
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary
from argparse import Namespace
import compresslab.nn
import compresslab.data
from pydantic_yaml import parse_yaml_file_as
import pickle
import compresslab.utils.registry
import importlib.util


class Args(Namespace):
    config: str = None
    list: bool = False
    test: bool = False
    cpu: bool = False

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
        datamodule.setup(None)

        exp_dir = os.path.join(config.Train.Output, Path(args.config).stem)
        os.makedirs(exp_dir, exist_ok=True)
        os.system(f"cp {args.config} {exp_dir}/config.yaml")

        for model in config.Model:
            out_dir = os.path.join(exp_dir, model.Name if model.Name is not None else model.Key) 
            
            # traditional codec
            if model.Key in TRADITIONAL_CODEC:
                codec = TRADITIONAL_CODEC[model.Key](
                    save_dir=out_dir,
                    **config.Data.Params,
                    **model.Params
                    )
                codec.run()
                continue


            # neural codec
            model_path = Path(getattr(compresslab.utils.registry, "ModelRegistry")._map.get(model.Key)["register_path"])

            module_file = model_path.parent / "trainer.py"
            if not module_file.exists():
                raise FileNotFoundError(f"trainer.py not found in {model_path.parent}")

            module_spec = importlib.util.spec_from_file_location("trainer", module_file)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)

            lightning_classes = [
                cls for name, cls in inspect.getmembers(module, inspect.isclass)
                if issubclass(cls, BaseTrainer) and cls.__module__ == module.__name__
            ]

            if not lightning_classes:
                raise ValueError(f"No class inherits from `BaseTrainer` found in {module_file}")

            assert len(lightning_classes) == 1, f"Found multiple Trainers in {module_file}. Please ensure only one class inherits from `BaseTrainer`."
            
            LightningModule = lightning_classes[0]
            
            modelmodule = LightningModule(model_class=ModelRegistry.get(model.Key), params=model.Params, ext_params=model.ExtParams)

            model_pkl_path = os.path.join(out_dir, "hparams.pkl")
            
            if not os.path.exists(model_pkl_path):
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "hparams.yaml"), "w") as f:
                    for attr, value in vars(model).items():
                        f.write(f"{attr}: {value}\n")
                
                # Save the model hyperparams as a pickle file
                with open(model_pkl_path, "wb") as pkl_file:
                    pickle.dump(model, pkl_file)
            else:
                with open(model_pkl_path, "rb") as pkl_file:
                    hparams = pickle.load(pkl_file)
                if hparams != model:
                    raise ValueError(f"Model hyperparams mismatch: {hparams} vs {model}")
            
            if config.Train.Steps is None and config.Train.Epoch is None:
                raise ValueError("Please specify either Train.Steps or Train.Epoch in the config file.")

            num_epoch = config.Train.Epoch if config.Train.Epoch is not None \
                else math.ceil(config.Train.Steps / len(datamodule.train_dataloader()))

            if not args.test:
                trainer = Trainer(
                    accelerator="gpu" if torch.cuda.is_available() else "cpu",
                    devices=config.Env.Devices,
                    strategy="ddp_find_unused_parameters_true",
                    max_steps=config.Train.Steps,
                    max_epochs=num_epoch,
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
                            max_depth=3,
                        )
                    ],
                    logger=TensorBoardLogger(save_dir=out_dir),
                    deterministic="warn" # NOTE: this is important for reproducibility, otherwise the entropy decoding may fail
                )
                trainer.fit(modelmodule, datamodule, ckpt_path="last")
            
            trainer = Trainer(
                    accelerator="cpu" if args.cpu else "gpu" if torch.cuda.is_available() else "cpu",
                    devices=1 if args.cpu else [config.Env.Devices[0]],
                    default_root_dir=out_dir,
                    callbacks=[RichProgressBar()],
                    logger=False,
                    deterministic="warn" # NOTE: this is important for reproducibility, otherwise the entropy decoding may fail
                )

            if os.path.exists(os.path.join(out_dir, f"checkpoints/mse.ckpt")):
                trainer.test(modelmodule, datamodule, ckpt_path=os.path.join(out_dir, f"checkpoints/mse.ckpt"))
            if os.path.exists(os.path.join(out_dir, f"checkpoints/ms-ssim.ckpt")):
                trainer.test(modelmodule, datamodule, ckpt_path=os.path.join(out_dir, f"checkpoints/ms-ssim.ckpt"))
            if not os.path.exists(os.path.join(out_dir, "checkpoints/mse.ckpt")) and not os.path.exists(os.path.join(out_dir, "checkpoints/ms-ssim.ckpt")):
                trainer.test(modelmodule, datamodule, ckpt_path="last")

        
        Benchmark(exp_dir, config.Train.Benchmark)
        
        logging.info("Finish training.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('-c', '--config', required=False, type=str,help ='Config file path.', default=None)
    parser.add_argument('-l', '--list', action="store_true", help='List all available models.')
    parser.add_argument('--test', action="store_true", help='Test only.')
    parser.add_argument('--cpu', action="store_true", help='Use CPU in the inference.')
    
    args = parser.parse_args()
    main(args)