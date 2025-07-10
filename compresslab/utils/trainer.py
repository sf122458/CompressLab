import os
import lightning as L
from compresslab.core.models import CompressionModel
from compresslab.utils.logger import MetricLogger
import torch
from typing import Dict, Any, Type
from compresslab.utils.wrapper import ModelWrapper

class BaseTrainer(L.LightningModule):
    """
    A trainer class inherits from PyTorch Lightning's LightningModule.
    """
    def __init__(self, 
                 model_class: Type[CompressionModel],
                 params: Dict[str, Any],
                 ext_params: Dict[str, Any] = None
                 ):
        """
        Args:
            model_class (Type[CompressionModel]): The model class to be trained.
            params (Dict[str, Any]): Parameters used in the model instantiation.
            ext_params (Dict[str, Any], optional): Additional parameters for the trainer.
                Defaults to None.
        """
        super().__init__()

        # As there are two optimizers, we need to set the automatic optimization to False
        self.automatic_optimization = False

        # compression levels
        if "lmbda" not in ext_params.keys():
            raise ValueError("key `lmbda` is required in `ext_params`")
        
        self._lmbda = ext_params.get("lmbda")

        if isinstance(self._lmbda, dict):
            assert "mse" in self._lmbda.keys() and "ms-ssim" in self._lmbda.keys()
            num_models = len(self._lmbda["mse"])
        else:
            if isinstance(self._lmbda, (int, float)):
                self._lmbda = [self._lmbda]
            else:
                assert isinstance(self._lmbda, list)
            num_models = len(self._lmbda)

        print(self._lmbda)

        self.model_wrapper = ModelWrapper(model_class, params, num_models)
        
        # optimization parameters
        self.lr = ext_params.get("lr", 1e-4)

        # some functions to log metrics
        self.bar_metrics = lambda metrics: self.log_dict(
            {f"{k}": v for k, v in metrics.items()},
            prog_bar=True, on_step=True, on_epoch=False, 
            logger=False, sync_dist=False, rank_zero_only=True
        )

        self.log_train_metrics = lambda model_name, metrics: self.log_dict(
            {f"train/{model_name}.{k}": v for k, v in metrics.items()},
            on_step=True, on_epoch=False, logger=True, 
            sync_dist=False, rank_zero_only=True
        )

        self.log_val_metrics = lambda model_name, metrics: self.log_dict(
            {f"val/{model_name}.{k}": v for k, v in metrics.items()},
            on_step=False, on_epoch=True, logger=True, sync_dist=True
        )

        self.log_test_metrics = lambda model_name, metrics: self.metric.log(model_name, metrics) \
            if self.metric is not None else self.log_dict(
            {f"test/{model_name}.{k}": v for k, v in metrics.items()},
            on_step=False, on_epoch=True, logger=True, sync_dist=True
        )
        
        self.log_train_monitor = lambda monitor: self.log_dict(
            {f"train_monitor/{k}": v for k, v in monitor.items()},
            on_step=False, on_epoch=True, logger=True, 
            sync_dist=False, rank_zero_only=True
        )

    @property
    def lmbda(self):
        if not isinstance(self._lmbda, dict):
            return self._lmbda
        
        if self.global_step < self.key_step["fine_tune"]:
            return self._lmbda["mse"]
        else:
            return self._lmbda["ms-ssim"]
        
    def on_train_start(self):
        self.key_step = {
            "lr_decay": int(self.trainer.max_steps * 0.4),
            "fine_tune": int(self.trainer.max_steps * 0.95) if isinstance(self._lmbda, dict) \
                else int(self.trainer.max_steps)
        }

    def on_train_batch_end(self, output, batch, batch_idx):
        """
        I think there are some operations can be done here:
        - Save the checkpoint of the model trained on mse-loss
        - Switch to fine-tune the model on ms-ssim-loss after a certain number of steps.
        - Adjust the learning rate or other hyperparameters if needed.
        - Multi stage training(Loss function modification)
        """
        pass

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Please implement the `training_step` method in your trainer class.")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("Please implement the `validation_step` method in your trainer class.")

    def on_test_start(self):
        if "mse" in self.trainer.ckpt_path:
            filename = "metrics_mse"
        elif "ms-ssim" in self.trainer.ckpt_path:
            filename = "metrics_ms_ssim"
        else:
            filename = "metrics"

        self.metric = MetricLogger(save_dir=self.trainer.default_root_dir,
                                   filename=filename)
        
        self.model_wrapper.update()

    def test_step(self, batch, batch_idx):
        raise NotImplementedError("Please implement the `test_step` method in your trainer class.")

    def on_test_end(self):
        """
        Save the metrics into a csv file and a pkl file.
        """
        self.metric.save()

    def configure_optimizers(self):
        parameters = []
        aux_parameters = []
        for model_instance in self.model_wrapper.values():
            parameters += [p for n, p in model_instance.named_parameters() if p.requires_grad and not n.endswith(".quantiles")]
            aux_parameters += [p for n, p in model_instance.named_parameters() if p.requires_grad and n.endswith(".quantiles")]
        
        optimizer = torch.optim.Adam([
            {"params": parameters, "lr": self.lr, "name": "model_params"},
            {"params": aux_parameters, "lr": 1e-3, "name": "entropy_model_params"}
        ])

        return optimizer