from compresslab.config import Config
from compresslab.utils.registry import OptimizerRegistry, SchedulerRegistry, CompoundRegistry, ModelRegistry, LossRegistry
from compresslab.data.dataset import Dataset, ImageDataset
from torch.utils.data import DataLoader
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TimeRemainingColumn
from typing import Dict, Any, Union
from wandb.sdk.wandb_run import Run
import datetime
import os
import glob

class _baseTrainer:
    def __init__(self, config: Config, run, resume: str=None):
        logging.info("Initialize the trainer...")
        # TODO: DDP
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # logging.info(f"Using device: {self.device}")

        self.config = config
        self.run = run

        self.start_epoch = 0
        self._step = 0

        self._set_dataloader()
        self._set_modules()
        if resume is not None:
            self._load_ckpt(resume)
        self.epoch = self.start_epoch

    def _set_dataloader(self):
        self.trainloader = DataLoader(
            ImageDataset(self.config.Train.TrainSet.Path, self.config.Train.TrainSet.Transform),
            batch_size=self.config.Train.BatchSize,
            shuffle=True,
            num_workers=self.config.ENV.NUM_WORKERS if self.config.ENV.NUM_WORKERS is not None else 0
        )

        self.valloader = DataLoader(
            ImageDataset(self.config.Train.ValSet.Path, self.config.Train.ValSet.Transform),
            # batch_size=config.Train.BatchSize,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.ENV.NUM_WORKERS if self.config.ENV.NUM_WORKERS is not None else 0
        )

    def _set_modules(self):
        self.compound = CompoundRegistry.get(self.config.Model.Compound)(self.config)
        self.optimizer = OptimizerRegistry.get(self.config.Train.Optim.Key)(self.compound.model.parameters(), **self.config.Train.Optim.Params)
        self.scheduler = SchedulerRegistry.get(self.config.Train.Schdr.Key)(self.optimizer, **self.config.Train.Schdr.Params)\
                        if self.config.Train.Schdr is not None else None

    def _load_ckpt(self, resume: str):
        ckpt = torch.load(resume)
        self.compound.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_epoch = ckpt["epoch"] + 1

    def _save_ckpt(self, add: Dict[str, Any]=None, overwrite=True):
        checkpoint = {
            "model": self.compound.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "next_epoch": self.epoch + 1
        }

        if add is not None:
            checkpoint.update(add)

        # whether to overwrite
        ckpt_list = [] if not overwrite else glob.glob(os.path.join(self.config.Train.Output, 'ckpt', '*.ckpt'))
        ckpt_name = f"ckpt/epoch_{self.epoch:0>3}_step_{self._step}.ckpt"
        torch.save(checkpoint, os.path.join(self.config.Train.Output, ckpt_name))
        if len(ckpt_list) > 0:
            for ckpt in ckpt_list:
                os.remove(ckpt)
        logging.info(f"Save checkpoint {ckpt_name}")

        
    def _beforeRun(self):
        self.progress = Progress(
            "[i blue]{task.description}[/][b magenta]{task.fields[progress]}", 
            TimeElapsedColumn(), 
            BarColumn(None), 
            TimeRemainingColumn(), 
            "{task.fields[suffix]}", 
            refresh_per_second=6, 
            transient=True, 
            disable=False, expand=True)
        self.progress.start()
        self.trainingBar = self.progress.add_task("", start=False, progress="[----/----]", suffix='.' * 10)
        self.progress.start_task(self.trainingBar)
        self.progress.update(
            self.trainingBar, 
            total=len(self.trainloader)*(self.config.Train.Epoch-self.start_epoch),
            completed=self._step,
            progress=f"[{self._step}/{len(self.trainloader)*(self.config.Train.Epoch-self.start_epoch):4d}]"
            )

    def _afterStep(self, log: Dict[str, Any], suffix: str, step_interval=10, **kwargs):
        if isinstance(self.run, Run) and self._step % step_interval == 0:
            self.run.log(log, step=self._step, **kwargs)

        self._step += 1
        self.progress.update(
            self.trainingBar, 
            advance=1, 
            progress=f"[{self._step}/{len(self.trainloader)*(self.config.Train.Epoch-self.start_epoch):4d}]", 
                suffix=suffix)

    def _afterEpoch(self):
        if (self.epoch + 1) % self.config.Train.ValInterval == 0 or self.epoch == self.config.Train.Epoch - 1:
            self.validate()
            self._save_ckpt()
        self.epoch += 1


    def train(self):
        raise NotImplementedError
    
    def validate(self):
        raise NotImplementedError
    
    def test(self):
        raise NotImplementedError

    
"""
    Compound is responsible for the feedforward, compression and decompression process of the model and the loss calculation.
    Components of the Compound class should include the backbone model, the loss function. These should be implemented in `__init__` method.
    In the `forward` method, the Compound class should return a dictionary with specific keys.
"""

import torch.nn as nn
import torch
import thop
import logging
from copy import deepcopy
from compresslab.loss import LossFn

class _baseCompound:
    def __init__(self, config: Config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        self.config = config
        self.model = ModelRegistry.get(config.Model.Net.Key)(**config.Model.Net.Params).to(self.device)
        self.loss = LossFn(config)

    def _paramsCalc(self, input=None):
        tensor = input if input is not None else torch.randn(1, 3, 256, 256).to(next(self.model.parameters()).device)
        # NOTE: thop will add `total_params` and `total_ops` to the model, so we need to deepcopy the model to avoid changing the original model
        flops, params = thop.profile(deepcopy(self.model), inputs=(tensor, ), report_missing=True)
        logging.info(f"FLOPs: {flops}, Params: {params}")
        
    
    def __call__(self, x: torch.Tensor):
        """
        Need to implement:
            Return should be a dictionary with the following keys
                - loss: total loss value
                - log: benchmark required to be recorded, e.g. PSNR, SSIM, etc.
            and other keys according to the task, such as x_hat, likelihoods, etc in image compression.
        """
        raise NotImplementedError
    
    # def inference(self, x: torch.Tensor):
    #     raise NotImplementedError
    
    # def compress(self, x: torch.Tensor):
    #     raise NotImplementedError
    
    # def decompress(self, x: torch.Tensor):
    #     raise NotImplementedError
    
from .logging_utils import MetricLogger, TorchCUDATimeProfiler
import functools

class _baseModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.profiler = MetricLogger()
    
    def start_time_profile(self, name, include_cpu_time=True, skip_cuda_synchronize=False):
        profiler_class = functools.partial(TorchCUDATimeProfiler, include_cpu_time=include_cpu_time, skip_cuda_synchronize=skip_cuda_synchronize)
        return self.profiler.start_time_profile(name, profiler_class=profiler_class)
        