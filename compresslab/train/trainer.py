"""
Trainer
    components:
        - dataloader
        - compound (model + loss)
        - optimizer
        - scheduler(optional)
    pipeline:
        1. Initialize compound, optimizer, and scheduler.
        2. In each step, the output of the model and the loss are obtained from the defined compound.
        3. Record the loss and the output of the model in the forward stage of the compound.
        4. Evaluate the model's performance of compression and decompression in the validation stage.
"""

import os
import glob
import logging
import torch
import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TimeRemainingColumn
from typing import Dict, Any, Union
from wandb.sdk.wandb_run import Run
from compresslab.config import Config
from compresslab.utils.registry import OptimizerRegistry, SchedulerRegistry, CompoundRegistry
from compresslab.data.dataset import Dataset, ImageDataset
from compresslab.utils.base import _baseTrainer

# Default trainer
class Trainer(_baseTrainer):
    def __init__(self, config: Config, run: Union[Run], resume: str=None):
        logging.info("Initialize Trainer.")
        # TODO: DDP
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Use {self.device}.")
        
        self.config = config

        self.trainloader = DataLoader(
            ImageDataset(config.Train.TrainSet.Path, config.Train.TrainSet.Transform),
            batch_size=config.Train.BatchSize,
            shuffle=True,
            num_workers=config.ENV.NUM_WORKERS if config.ENV.NUM_WORKERS is not None else 0
        )

        self.valloader = DataLoader(
            ImageDataset(config.Train.ValSet.Path, config.Train.ValSet.Transform),
            # batch_size=config.Train.BatchSize,
            batch_size=1,
            shuffle=False,
            num_workers=config.ENV.NUM_WORKERS if config.ENV.NUM_WORKERS is not None else 0
        )

        self.run = run
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

        self.compound = CompoundRegistry.get(config.Model.Compound)(config).to(self.device)
        self.optimizer = OptimizerRegistry.get(config.Train.Optim.Key)(self.compound.model.parameters(), **config.Train.Optim.Params)
        self.scheduler = SchedulerRegistry.get(config.Train.Schdr.Key)(self.optimizer, **config.Train.Schdr.Params) if config.Train.Schdr is not None else None
        self.start_epoch = 0

        if resume is not None:
            # ckpt = torch.load(resume, "cpu")
            ckpt = torch.load(resume)
            self.compound.load_state_dict(ckpt["net"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(ckpt["lr_scheduler"])
            self.start_epoch = ckpt["next_epoch"]


    def train(self):
        self._beforeRun()
        for epoch in range(self.config.Train.Epoch):
            for images in self.trainloader:
                images = images.to(self.device)
                # rewrite here if the output of the compound is different
                out = self.compound(images)
                
                # optimizer step
                out["loss"].backward()
                self.optimizer.zero_grad()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                #TODO: log: PSNR, SSIM, etc.

            if epoch % self.config.Train.ValInterval == 0:
                self.validate()

    def validate(self):
        with torch.no_grad():
            for images in self.valloader:
                images = images.to(self.device)
                out = self.compound(images)

    def _beforeRun(self):
        self.compound._paramsCalc()
        self._step = 0
        self.progress.start_task(self.trainingBar)
        self.progress.update(
            self.trainingBar, 
            total=len(self.trainloader)*self.config.Train.Epoch,
            completed=self._step,
            progress=f"[{self._step}/{len(self.trainloader)*self.config.Train.Epoch}]"
            )
        

    def _afterStep(self, **kwargs):
        self._step += 1
        self.progress.update(
            self.trainingBar, 
            advance=1, 
            progress=f"[{self._step}/{len(self.trainloader)*self.config.Train.Epoch:4d}]", 
                suffix=f"Bpp = [b green]{kwargs['bpp']:1.4f}, D = [b green]{kwargs['psnr']:2.2f}[/]dB")


# CompressAI models need to consider loss of nn models and aux loss of entropy models
class CompressAITrainer(Trainer):
    def __init__(self, config: Config, run: Union[Run], resume: str=None):
        logging.info("Initialize Trainer.")
        # TODO: DDP
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Use {self.device}.")
        
        self.config = config

        self.trainloader = DataLoader(
            ImageDataset(config.Train.TrainSet.Path, config.Train.TrainSet.Transform),
            batch_size=config.Train.BatchSize,
            shuffle=True,
            num_workers=config.ENV.NUM_WORKERS if config.ENV.NUM_WORKERS is not None else 0
        )

        self.valloader = DataLoader(
            ImageDataset(config.Train.ValSet.Path, config.Train.ValSet.Transform),
            # batch_size=config.Train.BatchSize,
            batch_size=1,
            shuffle=False,
            num_workers=config.ENV.NUM_WORKERS if config.ENV.NUM_WORKERS is not None else 0
        )

        self.run = run
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

        self.compound = CompoundRegistry.get(config.Model.Compound)(config).to(self.device)

        parameters = set(p for n, p in self.compound.model.named_parameters() if not n.endswith(".quantiles"))
        aux_parameters = set(p for n, p in self.compound.model.named_parameters() if n.endswith(".quantiles"))

        self.optimizer = OptimizerRegistry.get(config.Train.Optim.Key)(parameters, **config.Train.Optim.Params)
        self.aux_optimizer = OptimizerRegistry.get(config.Train.Optim.Key)(aux_parameters, **config.Train.Optim.Params)
        self.scheduler = SchedulerRegistry.get(config.Train.Schdr.Key)(optimizer=self.optimizer, **config.Train.Schdr.Params) if config.Train.Schdr is not None else None
        assert isinstance(self.scheduler, ReduceLROnPlateau)
        self.start_epoch = 0

        if resume is not None:
            ckpt = torch.load(resume, self.device)
            ckpt = torch.load(resume)
            self.compound.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(ckpt["lr_scheduler"])
            self.start_epoch = ckpt["next_epoch"]


    def train(self):
        self._beforeRun()
        for epoch in range(self.start_epoch, self.config.Train.Epoch):
            self.epoch = epoch
            for idx, images in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                self.aux_optimizer.zero_grad()
                images = images.to(self.device)

                out = self.compound(images)
                
                # optimizer step
                out["loss"].backward()
                
                self.optimizer.step()
                
                
                out["aux_loss"].backward()
                self.aux_optimizer.step()

                # if self.scheduler is not None:
                #     self.scheduler.step()

                #TODO: log: PSNR, SSIM, etc.
                
                self._afterStep(log=out["log"])
            
            if self.scheduler is not None:
                self.scheduler.step(self.validate())

            
            if (epoch + 1) % self.config.Train.ValInterval == 0 or epoch == self.config.Train.Epoch - 1:
                checkpoint = {
                    "model": self.compound.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "aux_optimizer": self.aux_optimizer.state_dict(),
                    "lr_scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                    "next_epoch": epoch + 1
                }

                # whether to overwrite
                time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                torch.save(checkpoint, os.path.join(self.config.Train.Output, 
                                                    f"ckpt/epoch_{epoch}_{time}.ckpt"))
                    
                    # if len(ckpt_list) > 0:
                    #     os.remove(os.path.join(ckpt_path, ckpt_list[0]))
                

    @torch.no_grad()
    def validate(self):
        bpp = 0
        psnr = 0
        loss = 0
        with torch.no_grad():
            for images in self.valloader:
                images = images.to(self.device)
                out = self.compound(images)
                loss += out["loss"]
                bpp += out["log"]["bpp"]
                psnr += out["log"]["psnr"]

        bpp /= len(self.valloader)
        psnr /= len(self.valloader)
        loss /= len(self.valloader)
        logging.info(f"Epoch {self.epoch}, validation result: Bpp = {bpp:1.4f}, PSNR = {psnr:2.2f}dB")
        return loss

    @torch.no_grad()
    def test(self):
        # TODO: Record decompressed images
        # TODO: Speed test
        # TODO: 
        pass



    def _afterStep(self, log: Dict[str, Any], **kwargs):
        if isinstance(self.run, Run):
            self.run.log(log, **kwargs)
        self._step += 1
        self.progress.update(
            self.trainingBar, 
            advance=1, 
            progress=f"[{self._step}/{len(self.trainloader)*self.config.Train.Epoch:4d}]",
            suffix=f"Bpp = [b green]{log['bpp']:1.4f}, D = [b green]{log['psnr']:2.2f}[/]dB")


    