import lightning as L
from compresslab.core.models import CompressionModel
from compresslab.utils.logger import MetricLogger
import torch
from typing import Dict, List, Any, Union, Type
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
        self.lmbda = ext_params.get("lmbda")
        if not isinstance(self.lmbda, list):
            self.lmbda = [self.lmbda]

        self.model_wrapper = ModelWrapper(model_class, params, len(self.lmbda))
        
        # optimization parameters
        self.lr = ext_params.get("lr", 1e-4)



        # some functions to log metrics
        self.bar_metrics = lambda metrics: self.log_dict(
            {f"{k}": v for k, v in metrics.items()},
            prog_bar=True, on_step=True, on_epoch=False, logger=False
        )

        self.log_train_metrics = lambda model_name, metrics: self.log_dict(
            {f"train/{model_name}.{k}": v for k, v in metrics.items()},
            on_step=True, on_epoch=False, logger=True, sync_dist=True
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
            on_step=True, on_epoch=False, logger=True, sync_dist=True
        )


    def on_train_batch_start(self, batch, batch_idx):
        """
        I think there are some operations can be done here:
        - Save the checkpoint of the model trained on mse-loss
        - Switch to fine-tune the model on ms-ssim-loss after a certain number of steps.
        - Adjust the learning rate or other hyperparameters if needed.
        """
        pass
    #     # switch to fine-tune the model after 1.5M steps to obtain the MS-SSIM model
    #     if self.global_step == self.trainer.max_steps:
    #         self.trainer.save_checkpoint(f"mse_{self.global_step}.ckpt")
    #         self.distortion = "ms-ssim"
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Please implement the `training_step` method in your trainer class.")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("Please implement the `validation_step` method in your trainer class.")

    def on_test_start(self):
        self.metric = MetricLogger(save_dir=self.trainer.default_root_dir,
                                   filename=self.trainer.ckpt_path.split(".")[0])
        # Only save the checkpoint before `update`, so it's required to call `update` before testing.
        for model_name, model_instance in self.model_wrapper.items():
            model_instance.update()

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
        for model_name, model_instance in self.model_wrapper.items():
            parameters += [p for n, p in model_instance.named_parameters() if p.requires_grad and not n.endswith(".quantiles")]
            aux_parameters += [p for n, p in model_instance.named_parameters() if p.requires_grad and n.endswith(".quantiles")]
        
        optimizer = torch.optim.Adam(parameters, lr=self.lr)
        aux_optimizer = torch.optim.Adam(aux_parameters, lr=1e-3) # follow the implementation in `CompressAI`
        
        # TODO
        return {
            "optimizer": [optimizer, aux_optimizer],
            # "lr_scheduler": 
        }