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
        out: q -> [batch_size, N_quantiles]
        label: g -> [batch_size, ].
        '''
        n = out.shape[1]
        batch_size = out.shape[0]
        # label = torch.ones_like(out) * label
        label = label.reshape((batch_size, 1)).repeat(1, n)
        tau = ((torch.arange(n) + .5 )/ n).repeat(batch_size, 1)
        d = label - out
        h = self.huber_loss(out, label)
        k = torch.abs(tau - (d < 0).float())
        loss = torch.mean(k * h)
        loss = Variable(loss, requires_grad=True)
        return loss