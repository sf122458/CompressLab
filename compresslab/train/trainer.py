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

import logging
import torch
from torch.utils.data import DataLoader
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TimeRemainingColumn
from compresslab.config.config import Config
from compresslab.utils.registry import OptimizerRegistry, SchedulerRegistry, CompoundRegistry, TrainerRegistry
from compresslab.dataset.data import Dataset, ImageDataset
from compresslab.utils.base import _baseTrainer

# Default trainer
@TrainerRegistry.register("Default")
class Trainer(_baseTrainer):
    def __init__(self, config: Config, run):
        logging.info("Initialize Trainer.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Use {self.device}.")
        
        self.config = config

        self.compound = CompoundRegistry.get(config.Model.Compound)(config).to(self.device)

        self.optimizer = OptimizerRegistry.get(config.Train.Optim.Key)(self.compound.model.parameters(), **config.Train.Optim.Params)
        self.scheduler = SchedulerRegistry.get(config.Train.Schdr.Key)(self.optimizer, **config.Train.Schdr.Params) if config.Train.Schdr is not None else None
        
        self.trainloader = DataLoader(
            Dataset(config.Train.TrainSet.Path, config.Train.TrainSet.Transform),
            batch_size=config.Train.BatchSize,
            shuffle=True,
            num_workers=config.ENV.NUM_WORKERS if config.ENV.NUM_WORKERS is not None else 0
        )

        self.valloader = DataLoader(
            Dataset(config.Train.ValSet.Path, config.Train.ValSet.Transform),
            # batch_size=config.Train.BatchSize,
            batch_size=1,
            shuffle=False,
            num_workers=config.ENV.NUM_WORKERS if config.ENV.NUM_WORKERS is not None else 0
        )

        self.run = run


    def train(self):
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
                self.val()

    def val(self):
        with torch.no_grad():
            for images in self.valloader:
                images = images.to(self.device)
                out = self.compound(images)

                

    def log(self):
        pass

# CompressAI models needs to consider loss of nn models and aux loss of entropy models
@TrainerRegistry.register("CompressAI")
class CompressAITrainer(_baseTrainer):
    def __init__(self, config: Config, run):
        
        logging.info("Initialize Trainer.")
        # TODO: DDP
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Use {self.device}.")
        
        self.config = config

        self.compound = CompoundRegistry.get(config.Model.Compound)(config).to(self.device)

        parameters = set(p for n, p in self.compound.model.named_parameters() if not n.endswith(".quantiles"))
        aux_parameters = set(p for n, p in self.compound.model.named_parameters() if n.endswith(".quantiles"))

        self.optimizer = OptimizerRegistry.get(config.Train.Optim.Key)(parameters, **config.Train.Optim.Params)
        self.aux_optimizer = OptimizerRegistry.get(config.Train.Optim.Key)(aux_parameters, **config.Train.Optim.Params)

        self.scheduler = SchedulerRegistry.get(config.Train.Schdr.Key)(self.optimizer, **config.Train.Schdr.Params) if config.Train.Schdr is not None else None
        
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
        

    def train(self):
        self._beforeRun()
        for epoch in range(self.config.Train.Epoch):
            for idx, images in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                self.aux_optimizer.zero_grad()
                images = images.to(self.device)

                # rewrite here if the output of the compound is different
                out = self.compound(images)
                
                # optimizer step
                out["loss"].backward()
                
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                
                out["aux_loss"].backward()
                self.aux_optimizer.step()

                if idx % 10 == 0:
                    print(out["loss"].item(), out["aux_loss"].item(), out["bpp_loss"].item(), out["log"]["psnr"])

                #TODO: log: PSNR, SSIM, etc.
                self._step += 1
                self._afterStep(psnr=out["log"]["psnr"])
            # if epoch % self.config.Train.ValInterval == 0:
            #     self.val()

    def val(self):
        with torch.no_grad():
            for images in self.valloader:
                images = images.to(self.device)
                out = self.compound(images)

                bpp = out["bpp_loss"].item()
                psnr = out["log"]["psnr"]

                

    def log(self):
        pass

    def _beforeRun(self):
        self._step = 0
        self.progress.start_task(self.trainingBar)
        self.progress.update(
            self.trainingBar, 
            total=len(self.trainloader)*self.config.Train.Epoch,
            completed=self._step,
            progress=f"[{self._step}/{len(self.trainloader)*self.config.Train.Epoch}]"
            )

    def _beforeStep(self):
        pass

    def _afterStep(self, **kwargs):
        # task = self.progress.get_task
        self.progress.update(
            self.trainingBar, 
            advance=1, 
            progress=f"[{self._step}/{len(self.trainloader)*self.config.Train.Epoch:4d}]", suffix=f"D = [b green]{kwargs['psnr']:2.2f}[/]dB")