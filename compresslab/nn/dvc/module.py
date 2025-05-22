import lightning as L
from typing import Union, List
from compressai.models import CompressionModel
import torch
import torch.nn as nn
from copy import deepcopy

class VideoLightingModule(L.LightningModule):
    def __init__(self,
                 model: CompressionModel,
                 lr: float = 1e-4,
                 lmbda: Union[float, List[float]] = 0.1,
                 ):
        super().__init__()

        self.automatic_optimization = False

        self.lmbda = lmbda
        self.lr = lr

        self.model_wrapper: nn.ModuleDict[str, CompressionModel] = nn.ModuleDict({})


        if isinstance(self.lmbda, list):
            for idx in range(len(self.lmbda)):
                self.model_wrapper[f"codec_{idx}"] = deepcopy(model)
            del model
        else:
            self.lmbda = [lmbda]
            self.model_wrapper["codec"] = model

    def training_step(self, batch, batch_idx):
        pass


    def validation_step(self, batch, batch_idx):
        pass
    

    def test_step(self, batch, batch_idx):
        pass
    

    def configure_optimizers(self):
        return super().configure_optimizers()