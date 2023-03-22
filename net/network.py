import torch
import torch.nn as nn


# Input:
#   Tensors of shape of [T,S,S,S]. First one is current tensor.
#   Scalars of shape of [s,].


class Torso(nn.Module):
    '''
    网络躯干(Encoder).
    '''
    
    def __init__(self,
                 S_size=4,
                 channel=3,
                 T=7,
                 scalar_size=3,
                 n_attentive=8,
                 **kwargs):
        super(Torso, self).__init__()
        self.channel = channel
        self.scalar_size = scalar_size
        self.T = T
        self.n_attentive = n_attentive
        self.attentive_modes = [AttentiveModes(S_size, channel) for _ in range(n_attentive)]
        self.scalar2grid = [nn.Linear(scalar_size, S_size**2) for _ in range(3)]                    # s -> S*S
        self.grid2grid = [nn.Linear(S_size**2*(T*S_size+1), S_size**2*channel) for _ in range(3)]   # S*S*TS+1 -> S*S*c
        
        
    def forward(self, x):
        S_size = self.S_size
        T = self.T
        channel = self.channel
        n_attentive = self.n_attentive
        
        input_t, input_s = x      # Tensor input and Scalar input.
        input_t, input_s = torch.from_numpy(input_t), torch.from_numpy(input_s)
        
        # 1. Project to grids.
        x1 = torch.reshape(torch.permute(input_t, (1,2,3,0)), (S_size, S_size, T*S_size))
        x2 = torch.reshape(torch.permute(input_t, (3,1,2,0)), (S_size, S_size, T*S_size))
        x3 = torch.reshape(torch.permute(input_t, (2,3,1,0)), (S_size, S_size, T*S_size))
        g = [x1, x2, x3]
        
        # 2. To grids.
        for idx in range(2):
            p = torch.reshape(self.scalar2grid[idx](input_s), (S_size, S_size, 1))
            g[idx] = torch.concat([g[idx], p], dim=-1)
            g[idx] = torch.reshape(self.grid2grid[idx](g[idx]), (S_size, S_size, channel))
            
        # 3. Attentive modes.
        x1, x2, x3 = g
        for idx in range(n_attentive):
            x1, x2, x3 = self.attentive_modes[idx]([x1, x2, x3])
            
        # 4. Final stack.
        e = torch.reshape(torch.stack([x1, x2, x3], axis=1), (3*S_size**2, channel))
        
        return e
        

class AttentiveModes(nn.Module):
    
    def __init__(self,
                 S_size=4,
                 channel=3):
        super(AttentiveModes, self).__init__()
        self.channel = channel
        self.S_size = S_size
        self.attentions = [Attention(channel) for _ in range(S_size)]
        
    def forward(self, x):
        
        for m1, m2 in [(0,1), (2,0), (1,2)]:
            a = torch.cat([x[m1], x[m2].transpose(0,1)], axis=1)
            ####### We are here! #######
                
    
class Attention(nn.Module):
    
    def __init__(self,
                 channel=3):
        super(Attention, self).__init__()
        pass
        

class PolicyHead():
    pass

class ValueHead():
    pass



class Net():
    '''
    网络部分
    '''
    
    def __init__(self,
                 **kwargs):
        '''
        初始化部分
        '''
        # e.g.:
        # self.n_layers = n_layers
        pass
    
    def policy_head(state, self):
        '''
        输出策略
        返回: actions, pi
        '''
        pass
    
    def value_head(state, self):
        '''
        输出效用值
        返回: reward
        '''
        pass