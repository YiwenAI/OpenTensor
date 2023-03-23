import numpy as np
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
        self.S_size = S_size
        self.channel = channel
        self.scalar_size = scalar_size
        self.T = T
        self.n_attentive = n_attentive
        
        self.attentive_modes = [AttentiveModes(S_size, channel) for _ in range(n_attentive)]
        self.scalar2grid = [nn.Linear(scalar_size, S_size**2) for _ in range(3)]                    # s -> S*S
        self.grid2grid = [nn.Linear(S_size**2*(T*S_size+1), S_size**2*channel) for _ in range(3)]   # S*S*TS+1 -> S*S*c
        
        
    def forward(self, x):
        
        # Input:
        #   Tensors of shape of [T,S,S,S]. First one is current tensor. (numpy)
        #   Scalars of shape of [s,].                                   (numpy)
        
        S_size = self.S_size
        T = self.T
        channel = self.channel
        n_attentive = self.n_attentive
        
        input_t, input_s = x      # Tensor input and Scalar input.
        input_t, input_s = torch.from_numpy(input_t).float(), torch.from_numpy(input_s).float()
        
        # 1. Project to grids.
        x1 = torch.reshape(torch.permute(input_t, (1,2,3,0)), (S_size, S_size, T*S_size))           # [S,S,TS]
        x2 = torch.reshape(torch.permute(input_t, (3,1,2,0)), (S_size, S_size, T*S_size))
        x3 = torch.reshape(torch.permute(input_t, (2,3,1,0)), (S_size, S_size, T*S_size))
        g = [x1, x2, x3]
        
        # 2. To grids.
        for idx in range(3):
            p = torch.reshape(self.scalar2grid[idx](input_s), (S_size, S_size, 1))
            g[idx] = torch.concat([g[idx], p], dim=-1)
            g[idx] = torch.reshape(self.grid2grid[idx](torch.reshape(g[idx], (-1,))), (S_size, S_size, channel))   # [S,S,c]
            
        # 3. Attentive modes.
        x1, x2, x3 = g
        for idx in range(n_attentive):
            x1, x2, x3 = self.attentive_modes[idx]([x1, x2, x3])
            
        # 4. Final stack.
        e = torch.reshape(torch.stack([x1, x2, x3], axis=1), (3*S_size**2, channel))
        
        return e
        

class AttentiveModes(nn.Module):
    
    '''
    问题：
        前向时, Attention模型是否共享参数?
    '''
    
    def __init__(self,
                 S_size=4,
                 channel=3):
        super(AttentiveModes, self).__init__()
        self.channel = channel
        self.S_size = S_size
        
        self.attentions = [Attention(channel,
                                     channel,
                                     2*S_size,
                                     2*S_size,
                                     False) for _ in range(S_size)]
        
    def forward(self, x):
        
        # Input:
        #   [x1, x2, x3]. Each of them is shaped of [S, S, c]
        
        S_size = self.S_size
        
        for m1, m2 in [(0,1), (2,0), (1,2)]:
            a = torch.cat([x[m1], x[m2].transpose(0,1)], axis=1)
            for idx in range(S_size):
                c = self.attentions[idx]([a[idx],])
                x[m1][idx] = c[:S_size, :]
                x[m2][idx] = c[S_size:, :]
        
        return x
            
    
class Attention(nn.Module):
    
    def __init__(self,
                 x_channel=3,
                 y_channel=3,
                 N_x=8,
                 N_y=8,
                 causal_mask=False,
                 N_heads=16,
                 d=32,
                 w=4):
        
        super(Attention, self).__init__()
        self.x_channel, self.y_channel = x_channel, y_channel
        self.N_x, self.N_y = N_x, N_y
        self.causal_mask = causal_mask
        self.N_heads = N_heads
        self.d, self.w = d, w
        
        self.x_layer_norm = nn.LayerNorm((N_x, x_channel))
        self.y_layer_norm = nn.LayerNorm((N_y, y_channel))
        self.final_layer_norm = nn.LayerNorm((N_x, x_channel))
        self.W_Q = nn.Linear(x_channel, d * N_heads)
        self.W_K = nn.Linear(y_channel, d * N_heads)
        self.W_V = nn.Linear(y_channel, d * N_heads)
        self.linear_1 = nn.Linear(d * N_heads, x_channel)
        self.linear_2 = nn.Linear(x_channel, x_channel * w)
        self.linear_3 = nn.Linear(x_channel * w, x_channel)
        self.gelu = nn.GELU()
        
        
    def forward(self, x):
        
        # Input:
        #   [x, (y)]. If y is missed, y=x. 
        
        N_heads = self.N_heads
        N_x, N_y = self.N_x, self.N_y
        d, w = self.d, self.w
        
        if len(x) == 1:
            x = x[0]
            y = x.clone()
        else:
            x, y = x
        
        x_norm = self.x_layer_norm(x)     # [N_x, c_x]
        y_norm = self.y_layer_norm(y)     # [N_y, c_y]
        
        q_s = self.W_Q(x_norm).view(-1, N_heads, d).transpose(0, 1)   # [N_heads, N_x, d]
        k_s = self.W_K(y_norm).view(-1, N_heads, d).transpose(0, 1)   # [N_heads, N_y, d]
        v_s = self.W_V(y_norm).view(-1, N_heads, d).transpose(0, 1)   # [N_heads, N_y, d]
        
        scores = torch.matmul(q_s, k_s.transpose(-1, -2)) / np.sqrt(d)    # [N_heads, N_x, N_y]
        if self.causal_mask:
            mask = np.triu(np.ones([N_heads, N_x, N_y]), k=1)
            scores = scores * mask
            
        o_s = torch.matmul(scores, v_s)   # [N_heads, N_x, d]
        o_s = o_s.transpose(0, 1).contiguous().view(-1, N_heads*d)    # [N_x, N_heads*d]
        x = x + self.linear_1(o_s)                                    # [N_x, c_x]
        
        x = x + self.linear_3(self.gelu(self.linear_2(self.final_layer_norm(x))))
        
        return x
        

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
    
    
    
if __name__ == '__main__':
    torso = Torso()
    test_input = [np.random.randint(-1, 1, (7, 4, 4, 4)), np.random.randint(-1, 1, (3,))]
    test_output = torso(test_input)
    import pdb; pdb.set_trace()