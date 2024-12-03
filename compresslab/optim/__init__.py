import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from compresslab.utils.registry import OptimizerRegistry, SchedulerRegistry

# Optimizer

OptimizerRegistry.register("Adam")(optim.Adam)

OptimizerRegistry.register("AdamW")(optim.AdamW)

OptimizerRegistry.register("SGD")(optim.SGD)

OptimizerRegistry.register("RMSprop")(optim.RMSprop)


# LrScheduler

SchedulerRegistry.register("StepLR")(lr_scheduler.StepLR)

SchedulerRegistry.register("MultiStepLR")(lr_scheduler.MultiStepLR)

SchedulerRegistry.register("ExponentialLR")(lr_scheduler.ExponentialLR)

SchedulerRegistry.register("CosineAnnealingLR")(lr_scheduler.CosineAnnealingLR)

SchedulerRegistry.register("ReduceLROnPlateau")(lr_scheduler.ReduceLROnPlateau)