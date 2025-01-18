"""
Trainer
    components:
        - dataloader
        - compound (model + loss)
        - optimizer
        - scheduler (optional)
    pipeline:
        1. Initialize compound, optimizer, and scheduler.
        2. In each step, the defined compound will output loss, data needed to log and other contents.
        3. Record the loss and the output of the model in the forward stage of the compound.
        4. Evaluate the model's performance of compression and decompression in the validation stage.
"""

import os
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
from torchvision.utils import save_image
import glob

# Default trainer
class Trainer(_baseTrainer):
    def train(self):
        self._beforeRun()
        for _ in range(self.start_epoch, self.config.Train.Epoch):
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

                self._afterStep(**out)
            self._afterEpoch()

    def validate(self):
        with torch.no_grad():
            for images in self.valloader:
                images = images.to(self.device)
                out = self.compound(images)
    
    def test(self):
        pass
        

    def _afterStep(self, log: Dict[str, Any], **kwargs):
        suffix = f"[b green]"
        for key, value in log.items():
            suffix += f"{key} = {value:.4f}, "
            
        super()._afterStep(log=log, suffix=suffix, **kwargs)

# CompressAI models need to consider loss of nn models and aux loss of entropy models
class CompressAITrainer(_baseTrainer):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, **kwargs)

        self.clip_max_norm = 1.0
        self.compound.model.train()

    def _set_modules(self, **kwargs):
        self.compound = CompoundRegistry.get(self.config.Model.Compound)(config=self.config, **kwargs)

        # DO NOT use set like the compressai implementation, which will mess up the optimizer. USE LIST.
        parameters = [p for n, p in self.compound.model.named_parameters() if p.requires_grad and not n.endswith(".quantiles")]
        aux_parameters = [p for n, p in self.compound.model.named_parameters() if p.requires_grad and n.endswith(".quantiles")]
        self.optimizer = OptimizerRegistry.get(self.config.Train.Optim.Key)(parameters, **self.config.Train.Optim.Params)
        self.aux_optimizer = OptimizerRegistry.get(self.config.Train.Optim.Key)(aux_parameters, lr=1e-3)
        self.scheduler = SchedulerRegistry.get(self.config.Train.Schdr.Key)(optimizer=self.optimizer, **self.config.Train.Schdr.Params)\
              if self.config.Train.Schdr is not None else None
        assert isinstance(self.scheduler, (ReduceLROnPlateau, None)), "Only ReduceLROnPlateau scheduler is supported."

    def _load_ckpt(self, model_name, **kwargs):
        self.ckpt_path = os.path.join(self.config.Train.Output, model_name, 'ckpt')
        os.makedirs(self.ckpt_path, exist_ok=True)
        ckpt_list = glob.glob(os.path.join(self.ckpt_path, '*.ckpt'))
        if len(ckpt_list) == 0:
            logging.info("No checkpoint found. Start training from the beginning.")
            return
        else:
            resume = sorted(ckpt_list)[-1]
            logging.info(f"Resume from {resume}")
            ckpt = torch.load(resume)
            self.compound.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            self.start_epoch = ckpt["next_epoch"] + 1


    def train(self):
        self._beforeRun()
        for _ in range(self.start_epoch, self.config.Train.Epoch):
            self.compound.model.train()
            for _, images in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                self.aux_optimizer.zero_grad()
                images = images.to(self.device)

                out = self.compound(images)
                
                # optimizer step
                out["loss"].backward()
                if self.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.compound.model.parameters(), self.clip_max_norm)
                
                self.optimizer.step()
                out["aux_loss"].backward()
                self.aux_optimizer.step()
                
                self._afterStep(out["log"])
            self._afterEpoch()
        self._afterRun()

                
    @torch.no_grad()
    def validate(self):
        bpp = 0
        psnr = 0
        loss = 0
        self.compound.model.eval()
        with torch.no_grad():
            for images in self.valloader:
                images = images.to(self.compound.device)
                out = self.compound(images)
                loss += out["loss"]
                bpp += out["log"]["bpp"]
                psnr += out["log"]["psnr"]

        bpp /= len(self.valloader)
        psnr /= len(self.valloader)
        loss /= len(self.valloader)
        logging.info(f"Epoch {self.epoch}, validation result: Loss = {loss:.4f}, Bpp = {bpp:1.4f}, PSNR = {psnr:2.2f}dB")
        return dict(
            loss=loss,
            bpp=bpp,
            psnr=psnr
        )

    @torch.no_grad()
    def test(self):
        # TODO: Record decompressed images
        # TODO: Speed test
        # TODO: 
        out = self.validate()
        with open(os.path.join(self.config.Train.Output, 'result.txt'), 'a') as f:
            f.write(f"Model: {self.model_name} Bpp = {out['bpp']:1.4f}, PSNR = {out['psnr']:2.2f}dB\n")
            f.close()

    def _afterStep(self, log: Dict[str, Any], **kwargs):
        suffix = f"[b green]Bpp = {log['bpp']:1.4f}, D = {log['psnr']:2.2f}dB"
        super()._afterStep(log=log, suffix=suffix, **kwargs)

    def _afterEpoch(self):
        if self.scheduler is not None:
            self.scheduler.step(self.validate()["loss"])
        if (self.epoch + 1) % self.config.Train.Valinterval == 0 or self.epoch == self.config.Train.Epoch - 1:
            self.validate()
            self._save_ckpt(add={"aux_optimizer": self.aux_optimizer.state_dict()})
        self.epoch += 1
