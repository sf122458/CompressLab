from .trainer import *
from ..utils.registry import TrainerRegistry

TrainerRegistry.register("Default")(Trainer)
TrainerRegistry.register("CompressAI")(CompressAITrainer)