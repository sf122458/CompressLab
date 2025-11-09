import torch
import torch.nn as nn
import torch.nn.functional as F

class sigmoid_loss(torch.nn.Module):
    def __init__(self, alpha=1.):
        super().__init__()
        self.lossfn = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha

    def forward(self, input, for_real):
        if for_real:
            target = self.alpha*torch.tensor(1.)
        else:
            target = torch.tensor(0.)

        target_ = target.expand_as(input).to(input.device)
        loss = self.lossfn(input, target_).mean(1).reshape(-1, 1)
        return loss

class hinge_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, for_real=True, for_G=False):
        if for_G:
            loss = -torch.mean(input)
        else:
            if for_real:
                loss = torch.mean(F.relu(1. - input))
            else:
                loss = torch.mean(F.relu(1. + input))
                    
        return loss