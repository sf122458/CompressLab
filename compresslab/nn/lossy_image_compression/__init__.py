from compresslab.utils.registry import ModelRegistry
from .compressai_impl.models import *
from .mlic.models import MLICPlusPlus
from .charm.models import ChARM

for model in [
    FactorizedPrior,
    ScaleHyperprior,
    MeanScaleHyperprior,
    JointAutoregressiveHierarchicalPriors,
    ChARM,
    Cheng2020Attention,
    Cheng2020Anchor,
    Cheng2020AnchorCheckerboard,
    Elic2022Official,
    Elic2022Chandelier,
    MLICPlusPlus
]:
    ModelRegistry.register(model.__name__)(model)