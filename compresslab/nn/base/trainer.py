import lightning as L
from compresslab.nn.base.utils import write_body, read_body, filesize
from compresslab.core.models import CompressionModel, update_registered_buffers
from compresslab.core.entropy_models import EntropyBottleneck, GaussianConditional
from compresslab.utils.logger import MetricLogger
import torch, os
from torch import Tensor
from pathlib import Path
import torch.nn as nn
from typing import Dict, Any, Type
from .wrapper import ModelWrapper
from typing import List, Tuple
from compresslab.utils.config import (
    GeneralCodecExtParams
)
from compresslab.nn.base.metrics import MetricsCollector
from torchvision.utils import save_image

class BasicTrainer(L.LightningModule):
    """
    A trainer class inherits from PyTorch Lightning's LightningModule.
    """
    def __init__(self, ext_params: GeneralCodecExtParams):
        super().__init__()

        # Set automatic optimization to False, as we will handle it manually
        self.automatic_optimization = False
        
        self.ext_params = ext_params
        self.metric_logger = MetricLogger()

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

        self.log_test_metrics = lambda model_name, metrics: self.metric_logger.log(model_name, metrics) \
            if self.metric_logger is not None else self.log_dict(
            {f"test/{model_name}.{k}": v for k, v in metrics.items()},
            on_step=False, on_epoch=True, logger=True, sync_dist=True
        )
        
        self.log_train_monitor = lambda monitor: self.log_dict(
            {f"train_monitor/{k}": v for k, v in monitor.items()},
            on_step=False, on_epoch=True, logger=True, 
            sync_dist=True, rank_zero_only=True
        )
        
        
        # metrics collector
        self.metrics_collector = MetricsCollector()

    def on_train_batch_end(self, output, batch, batch_idx):
        """
        There are some operations can be implemented here:
        - Save the checkpoint of the model trained on mse-loss.
        - Switch to fine-tune the model on ms-ssim-loss after a certain number of steps.
        - Adjust the learning rate or other hyperparameters if needed.
        - Multi stage training (Loss function modification).
        """
        pass

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Please implement the `training_step` method in your trainer class.")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("Please implement the `validation_step` method in your trainer class.")

    def on_test_start(self):
        self.metric_logger.reset_dir_and_filename(
            save_dir=self.trainer.default_root_dir,
        )

    def test_step(self, batch, batch_idx):
        raise NotImplementedError("Please implement the `test_step` method in your trainer class.")

    def on_test_end(self):
        """
        Save the metrics into a csv file and a pkl file.
        """
        self.metric_logger.save()

    def configure_optimizers(self):
        raise NotImplementedError("Please implement the `configure_optimizers` method in your trainer class.")

    def write_bitstream(self, filename: str, strings: List[List[bytes]], shape: Tuple[int, int]):
        bitstream_dir = os.path.join(
            self.trainer.default_root_dir,
            "bitstreams")
        os.makedirs(bitstream_dir, exist_ok=True)
        stream_path = os.path.join(bitstream_dir, f"{filename}.bin")
        with Path(stream_path).open("wb") as f:
            write_body(f, shape, strings)
        size = filesize(stream_path)
        return float(size) * 8
                
    def read_bitstream(self, filename: str):
        with Path(os.path.join(
            self.trainer.default_root_dir,
            "bitstreams",
            f"{filename}.bin"
        )).open("rb") as f:
            return read_body(f)
    
    def save_recon_imgs(self, imgs: Tensor, filename: str):
        output_dir = os.path.join(
            self.trainer.default_root_dir, 
            "recon_imgs",
        )
        os.makedirs(output_dir, exist_ok=True)
        if not filename.endswith(".png"):
            filename += ".png"
        save_image(imgs, os.path.join(output_dir, filename))
        
    
    def load_state_dict(self, state_dict, strict = True, assign = False):
        for name, module in self.named_modules():
            if not any(x.startswith(name) for x in state_dict.keys()):
                continue

            if isinstance(module, EntropyBottleneck):
                update_registered_buffers(
                    module,
                    name,
                    ["_quantized_cdf", "_offset", "_cdf_length"],
                    state_dict,
                    policy="resize"
                )

            if isinstance(module, GaussianConditional):
                update_registered_buffers(
                    module,
                    name,
                    ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                    state_dict,
                    policy="resize"
                )

        return nn.Module.load_state_dict(self, state_dict, strict=strict)


class CompressAIImageCodecTrainer(BasicTrainer):
    """This is used in the training for end-to-end lossy image compression.
    """
    def __init__(self, 
                 model_class: Type[CompressionModel],
                 params: Dict[str, Any],
                 *args, 
                 **kwargs
                 ):
        """
        Args:
            model_class (Type[CompressionModel]): The model class to be trained.
            params (Dict[str, Any]): Parameters used in the model instantiation.
            ext_params (Dict[str, Any], optional): Additional parameters for the trainer.
                Defaults to None.
        """
        super().__init__(*args, **kwargs)

        self.automatic_optimization = False

        # compression levels
        self._lmbda = self.ext_params.Lmbda

        if isinstance(self._lmbda, dict):
            assert "mse" in self._lmbda.keys() and "ms-ssim" in self._lmbda.keys()
            num_models = len(self._lmbda["mse"])
        else:
            if isinstance(self._lmbda, (int, float)):
                self._lmbda = [self._lmbda]
            else:
                assert isinstance(self._lmbda, list)
            num_models = len(self._lmbda)

        self.model_wrapper = ModelWrapper(model_class, params, num_models)
        

    def on_train_start(self):
        # FIXME: `self.finetune_step` must be defined here due to `Trainer` isn't attached before this step.
        # fine-tuning steps
        if isinstance(self._lmbda, dict) and "ms-ssim" in self._lmbda.keys():
            self.finetune_step = self.ext_params.FinetuneStep if self.ext_params.FinetuneStep > 0 else \
                int(self.ext_params.FinetuneRatio * self.trainer.max_steps) + 1 # when `FinetuneRatio` is 1, ms-ssim model won't be saved
        else:
            self.finetune_step = self.trainer.max_steps + 1

    @property
    def lmbda(self):
        if not isinstance(self._lmbda, dict):
            return self._lmbda
        
        if self.global_step < self.finetune_step:
            return self._lmbda["mse"]
        else:
            return self._lmbda["ms-ssim"]

    def on_train_batch_end(self, output, batch, batch_idx):
        """
        There are some operations can be implemented here:
        - Save the checkpoint of the model trained on mse-loss.
        - Switch to fine-tune the model on ms-ssim-loss after a certain number of steps.
        - Adjust the learning rate or other hyperparameters if needed.
        - Multi stage training (Loss function modification).
        """
        pass
    
    def on_test_start(self):
        self.model_type = "last"
        if self.trainer.ckpt_path is not None:
            if "mse" in self.trainer.ckpt_path:
                self.model_type = "mse"
            elif "ms_ssim" in self.trainer.ckpt_path:
                self.model_type = "ms_ssim"

        self.metric_logger.reset_dir_and_filename(
            save_dir=self.trainer.default_root_dir,
            filename=f"metrics_{self.model_type}"
        )
        
        self.model_wrapper.update()
        
    def configure_optimizers(self):
        parameters = []
        aux_parameters = []
        for model_instance in self.model_wrapper.values():
            parameters += [p for n, p in model_instance.named_parameters() if p.requires_grad and not n.endswith(".quantiles")]
            aux_parameters += [p for n, p in model_instance.named_parameters() if p.requires_grad and n.endswith(".quantiles")]
        
        optimizer = torch.optim.Adam([
            {"params": parameters, "lr": self.ext_params.Lr, "name": "model_params"},
            {"params": aux_parameters, "lr": self.ext_params.Auxlr, "name": "entropy_model_params"}
        ])

        return optimizer
    
# TODO
class VQCodecTrainer(BasicTrainer):
    pass


class LosslessImageCodec(BasicTrainer):
    pass


class OverfitImageCodec(BasicTrainer):
    pass