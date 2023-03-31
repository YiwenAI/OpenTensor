import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class CrossEntropyLoss(nn.Module):
    pass


class QuantileLoss(nn.Module):
    
    def __init__(self,
                 delta=1):
        super(QuantileLoss, self).__init__()
        self.delta = delta
        self.huber_loss = nn.HuberLoss(delta=delta, reduction="none")
        
    def forward(self, out, label) -> Variable:
        '''
        out: q -> [N_quantiles]
        label: g -> scalar.
        '''
        n = out.shape[0]
        label = torch.ones_like(out) * label
        tau = torch.arange(n) + .5 / n
        d = label - out
        h = self.huber_loss(out, label)
        k = torch.abs(tau - (d < 0).float())
        loss = torch.mean(k * h)
        loss = Variable(loss, requires_grad=True)
        return loss