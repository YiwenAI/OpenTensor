import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CrossEntropyLoss(nn.Module):
    pass


class QuantileLoss(nn.Module):
    
    def __init__(self,
                 delta=1,
                 device='cuda'):
        super(QuantileLoss, self).__init__()
        self.delta = delta
        self.device = device
        
    def forward(self, out, label) -> Variable:
        '''
        out: q -> [batch_size, N_quantiles]
        label: g -> [batch_size, ].
        '''
        n = out.shape[1]
        batch_size = out.shape[0]
        # label = torch.ones_like(out) * label
        label = label.reshape((batch_size, 1)).repeat(1, n).to(self.device)
        tau = ((torch.arange(n) + .5 )/ n).repeat(batch_size, 1).to(self.device)
        d = label - out
        h = F.huber_loss(out, label, reduction='none', delta=self.delta)
        k = torch.abs(tau - (d < 0).float())
        loss = torch.mean(k * h)
        return loss