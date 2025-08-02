from typing import Any, Dict, List, Sequence, Tuple, Union, Type
import torch
from torch import nn, Tensor
from torch.func import functional_call
from torch import vmap
from copy import deepcopy
from compresslab.core.models import CompressionModel

def stack_module_state(
    models: Union[Sequence[nn.Module], nn.ModuleList],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """stack_module_state(models) -> params, buffers

    Prepares a list of torch.nn.Modules for ensembling with :func:`vmap`.

    Given a list of ``M`` ``nn.Modules`` of the same class, returns two dictionaries
    that stack all of their parameters and buffers together, indexed by name.

    **Attention: The stacked parameters are related to the original parameters, which is 
    different from `torch.func.stack_module_state`.**
    """
    if len(models) == 0:
        raise RuntimeError("stack_module_state: Expected at least one model, got 0.")
    if not (all(m.training for m in models) or all(not m.training for m in models)):
        raise RuntimeError(
            "stack_module_state: Expected all models to have the same training/eval mode."
        )
    model0_typ = type(models[0])
    if not all(type(m) == model0_typ for m in models):
        raise RuntimeError(
            "stack_module_state: Expected all models to be of the same class."
        )
    all_params = [dict(model.named_parameters()) for model in models]
    params = {
        k: construct_stacked_leaf(tuple(params[k] for params in all_params), k)
        for k in all_params[0]
    }
    all_buffers = [dict(model.named_buffers()) for model in models]
    buffers = {
        k: construct_stacked_leaf(tuple(buffers[k] for buffers in all_buffers), k)
        for k in all_buffers[0]
    }

    return params, buffers


def construct_stacked_leaf(
    tensors: Union[Tuple[Tensor, ...], List[Tensor]], name: str
) -> Tensor:
    all_requires_grad = all(t.requires_grad for t in tensors)
    none_requires_grad = all(not t.requires_grad for t in tensors)
    if not all_requires_grad and not none_requires_grad:
        raise RuntimeError(
            f"Expected {name} from each model to have the same .requires_grad"
        )
    result = torch.stack(tensors)
    return result


class ModelWrapper(nn.Module):
    """
    ModelWrapper is a wrapper for a list of models.
    It support a vmap forward pass, which allows a parallel forward pass of multiple models.
    It also organizes the models in a ModuleList and ModuleDict format.
    """
    def __init__(self, model_class: Type[nn.Module], params: Dict[str, Any], num_models: int):
        super().__init__()

        for k, v in params.items():
            if not isinstance(v, list):
                params[k] = [v] * num_models
            else:
                assert len(v) == num_models, f"Parameter {k} should have length {num_models}."

        self.models = nn.ModuleList(
            [model_class(**{k: v[i] for k, v in params.items()}) for i in range(num_models)]
        )

        def fmodel(params, buffers, x):
            return functional_call(deepcopy(self.models[0]), (params, buffers), (x,))

        self.vmap_forward = vmap(fmodel, in_dims=(0, 0, None), randomness="different")
    
    def keys(self):
        return [f"codec_{idx}" for idx in range(len(self.models))]
    
    def values(self):
        return self.models
    
    def items(self):
        return {f"codec_{idx}": model for idx, model in enumerate(self.models)}.items()
    
    # CompressAI feature
    def _assert_compressmodel(self):
        assert isinstance(self.models[0], CompressionModel), \
            "ModelWrapper only supports `CompressionModel` instances."

    def update(self):
        self._assert_compressmodel()
        for model in self.models:
            model.update()

    def aux_loss(self) -> torch.Tensor:
        """
        This function sums the auxiliary loss for all models.
        """
        self._assert_compressmodel()
        aux_loss = 0.0
        for model in self.models:
            aux_loss += model.aux_loss()
        return aux_loss
    
    def forward(self, input):
        """
        vmap forward pass for all models, only supports `CompressionModel`.
        
        warning: `vmap` only supports the `forward` method with tensor inputs and tensor outputs.
        """
        self._assert_compressmodel()
        param, buffer = stack_module_state(self.models)
        return self.vmap_forward(param, buffer, input)