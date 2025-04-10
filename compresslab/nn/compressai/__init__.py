from compresslab.utils.registry import ModelRegistry
from .models import *

for model in [
    FactorizedPrior,
    ScaleHyperprior,
    MeanScaleHyperprior,
    JointAutoregressiveHierarchicalPriors,
    Cheng2020Attention,
    Cheng2020Anchor,
    Cheng2020AnchorCheckerboard,
    Elic2022Official,
    Elic2022Chandelier
]:
    ModelRegistry.register(model.__name__)(model)