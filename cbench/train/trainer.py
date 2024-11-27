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
from cbench.config.config import Config
from cbench.utils.registry import OptimizerRegistry, SchedulerRegistry, CompoundRegistry, TrainerRegistry
from cbench.dataset.data import Dataset

class _baseTrainer:
    def __init__(self, config: Config, run):
        logging.info("Initialize Trainer.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Use {self.device}.")
        
        self.config = config

        self.compound = CompoundRegistry.get(config.Model.Compound)(config)

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
        raise NotImplementedError
    
    def val(self):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError

@TrainerRegistry.register("Default")
class Trainer(_baseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


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