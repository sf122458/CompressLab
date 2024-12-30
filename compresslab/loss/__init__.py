from .loss import *

# default loss for image reconstruction, its input should be xHat and x
class LossFn(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.loss = []
        logging.debug(config.Model.Loss)
        for type, params in config.Model.Loss.items():
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