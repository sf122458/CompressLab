from .loss import *
from typing import Dict, Any

# default loss for image reconstruction, its input should be xHat and x
class LossFn(nn.Module):
    def __init__(self, loss_config: Dict[str, Any]):
        super().__init__()
        self.loss = []
        logging.debug(loss_config)
        for type, params in loss_config.items():
            try:
                loss_fn = LossRegistry.get(type)(**params)
                self.loss.append(loss_fn)
                logging.info(f"Register loss function: {type} with params: {params}")
            except:
                if type.upper() == "BPP":
                    logging.info(f"Find bpp loss. Please check the compound has implemented the bpp loss calculation.")
                else:
                    logging.warning(f"Loss function {type} is not implemented. Skip this loss function.")
        
        assert len(self.loss) > 0, "No loss function is registered."

    def forward(self, xHat, x, **kwargs):
        loss = 0
        for loss_fn in self.loss:
            loss += loss_fn(xHat, x)
        return loss