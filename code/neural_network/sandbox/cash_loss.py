import torch.nn as nn
import torch

class CashLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CashLoss, self).__init__()
        self.eps = eps

    def forward(self, predicted, target):
        # Ensure that the model has positive values for logarithm calculation,
        # add epsilon to avoid log(0)
        predicted = torch.clamp(predicted, min=self.eps)
        target += self.eps
        loss = 2 * (predicted - target + target * (torch.log(target) - torch.log(predicted)))
        return torch.sum(loss)
