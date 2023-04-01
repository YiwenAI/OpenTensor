import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys
import os
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join(".")))
from env import *
from mcts import *
from utils import *

# Input:
#   Tensors of shape of [B,T,S,S,S]. First one is current tensor.
#   Scalars of shape of [B,s].


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
                 mode="train",
                 **kwargs):
        super(Torso, self).__init__()
        self.S_size = S_size
        self.channel = channel
        self.scalar_size = scalar_size
        self.T = T
        self.n_attentive = n_attentive
        self.mode = mode
        
        self.attentive_modes = [AttentiveModes(S_size, channel) for _ in range(n_attentive)]
        self.scalar2grid = [nn.Linear(scalar_size, S_size**2) for _ in range(3)]                    # s -> S*S
        self.grid2grid = [nn.Linear(S_size**2*(T*S_size+1), S_size**2*channel) for _ in range(3)]   # S*S*TS+1 -> S*S*c
        
        
    def forward(self, x):
        
        # Input:
        #   Tensors of shape of [B,T,S,S,S]. First one is current tensor. (numpy)
        #   Scalars of shape of [B,s].                                    (numpy)
        
        S_size = self.S_size
        T = self.T
        channel = self.channel
        n_attentive = self.n_attentive
        
        input_t, input_s = x      # Tensor input and Scalar input.
        input_t, input_s = torch.from_numpy(input_t).float(), torch.from_numpy(input_s).float()
        if self.mode == "infer":
            assert len(input_t.shape) == 4 and len(input_s.shape) == 1, \
                "Infer mode does not support batch."
            input_t = input_t[None]; input_s = input_s[None]        # Add a batch dim.
        batch_size = input_t.shape[0]
        
        # 1. Project to grids.
        x1 = torch.reshape(torch.permute(input_t, (0,2,3,4,1)), (batch_size, S_size, S_size, T*S_size))           # [B,S,S,TS]
        x2 = torch.reshape(torch.permute(input_t, (0,4,2,3,1)), (batch_size, S_size, S_size, T*S_size))
        x3 = torch.reshape(torch.permute(input_t, (0,3,4,2,1)), (batch_size, S_size, S_size, T*S_size))
        g = [x1, x2, x3]
        
        # 2. To grids.
        for idx in range(3):
            p = torch.reshape(self.scalar2grid[idx](input_s), (batch_size, S_size, S_size, 1))
            g[idx] = torch.concat([g[idx], p], dim=-1)
            g[idx] = torch.reshape(self.grid2grid[idx](torch.reshape(g[idx], (batch_size, -1))), (batch_size, S_size, S_size, channel))   # [B,S,S,c]
            
        # 3. Attentive modes.
        x1, x2, x3 = g
        for idx in range(n_attentive):
            x1, x2, x3 = self.attentive_modes[idx]([x1, x2, x3])
            
        # 4. Final stack.
        e = torch.reshape(torch.stack([x1, x2, x3], axis=1), (batch_size, 3*S_size**2, channel))     # [B, 3*S**2, c]
        
        return e
    
    
    def set_mode(self, mode):
        assert mode in ["train", "infer"]
        self.mode = mode
        

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
        #   [x1, x2, x3]. Each of them is shaped of [B, S, S, c]
        
        S_size = self.S_size
        
        for m1, m2 in [(0,1), (2,0), (1,2)]:
            a = torch.cat([x[m1], x[m2].transpose(1,2)], axis=2)
            for idx in range(S_size):
                c = self.attentions[idx]([a[:,idx],])
                x[m1][:,idx] = c[:, :S_size, :]
                x[m2][:,idx] = c[:, S_size:, :]
        
        return x
            
    
class Attention(nn.Module):
    
    def __init__(self,
                 x_channel=3,
                 y_channel=3,
                 N_x=8,            # 2S
                 N_y=8,            # 2S
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
        
        self.x_layer_norm = nn.LayerNorm((x_channel,))
        self.y_layer_norm = nn.LayerNorm((y_channel,))
        self.final_layer_norm = nn.LayerNorm((x_channel,))
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
        
        batch_size = x.shape[0]
        
        x_norm = self.x_layer_norm(x)     # [B, N_x, c_x]
        y_norm = self.y_layer_norm(y)     # [B, N_y, c_y]
        
        q_s = self.W_Q(x_norm).view(batch_size, -1, N_heads, d).transpose(1, 2)   # [batch_size, N_heads, N_x, d]
        k_s = self.W_K(y_norm).view(batch_size, -1, N_heads, d).transpose(1, 2)   # [batch_size, N_heads, N_y, d]
        v_s = self.W_V(y_norm).view(batch_size, -1, N_heads, d).transpose(1, 2)   # [batch_size, N_heads, N_y, d]
        
        scores = torch.matmul(q_s, k_s.transpose(-1, -2)) / np.sqrt(d)    # [batch_size, N_heads, N_x, N_y]
        if self.causal_mask:
            mask = torch.from_numpy(np.triu(np.ones([batch_size, N_heads, N_x, N_y]), k=1)).float()
            scores = scores * mask
            
        o_s = torch.matmul(scores, v_s)   # [batch_size, N_heads, N_x, d]
        o_s = o_s.transpose(1, 2).contiguous().view(batch_size, -1, N_heads*d)          # [batch_size, N_x, N_heads*d]
        x = x.reshape(batch_size*N_x, -1) + self.linear_1(o_s.reshape(-1, N_heads*d))   # [batch_size*N_x, c_x]
        
        x = (x + self.linear_3(self.gelu(self.linear_2(self.final_layer_norm(x))))).reshape(batch_size, N_x, -1)     # [batch_size, N_x, c_x]
        
        return x
        

class PolicyHead(nn.Module):

    def __init__(self,
                 N_steps=6,        
                 N_logits=9,
                 N_features=64,
                 N_heads=32,
                 N_layers=2,
                 N_samples=32,
                 torso_feature_shape=(3*4**2, 3),
                 mode='train'):
        
        super(PolicyHead, self).__init__()
        self.N_steps = N_steps
        self.N_logits = N_logits
        self.N_features = N_features
        self.N_heads = N_heads
        self.N_layers = N_layers
        self.torso_feature_shape = torso_feature_shape
        self.N_samples = N_samples
        self.mode = mode
        
        self.linear_1 = nn.Linear(N_logits, N_features * N_heads)
        self.pos_embed = nn.Linear(1, N_features * N_heads)
        self.self_layer_norms = [nn.LayerNorm((N_features * N_heads)) for _ in range(N_layers)]
        self.cross_layer_norms = [nn.LayerNorm(( N_features * N_heads)) for _ in range(N_layers)]   #FIXME: How to choose layer norm's channel?
        self.self_attentions = [Attention(x_channel=N_features * N_heads,
                                          y_channel=N_features * N_heads,
                                          N_x=N_steps,
                                          N_y=N_steps,
                                          causal_mask=True,
                                          N_heads=N_heads) for _ in range(N_layers)]
        self.cross_attentions = [Attention(x_channel=N_features * N_heads,
                                           y_channel=torso_feature_shape[1],
                                           N_x=N_steps,
                                           N_y=torso_feature_shape[0],
                                           causal_mask=False,
                                           N_heads=N_heads) for _ in range(N_layers)]
        self.self_dropouts = [nn.Dropout() for _ in range(N_layers)]
        self.cross_dropouts = [nn.Dropout() for _ in range(N_layers)]
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(N_features * N_heads, N_logits)
        
    
    def forward(self, x):
        
        # Input:
        #   [e, (g)]. e is the features extracted by torso, g is groundtruth (available in train mode)
        #   e: [B, m, c]
        #   g: {0,1,... N_logits-1} ^ [B, N_steps], [B, N_steps]
        
        N_steps = self.N_steps
        N_logits = self.N_logits
        N_samples = self.N_samples
        assert self.mode in ['train', 'infer']
        
        if self.mode == 'train':
            e, g = x                            # g: {0,1,... N_logits-1} ^ [B, N_steps], [B, N_steps]
            g_onehot = one_hot(g, num_classes=N_logits).float()     # [B, N_steps, N_logits]
            #FIXME: We haven't applied "shift" operation.
            o, z = self.predict_action_logits(g_onehot, e)    # o: [B, N_steps, N_logits]; z: [B, N_steps, N_features*N_heads]
            return o, z[:, 0]                   # o: [B, N_steps, N_logits]; z[:, 0]: [B, N_features*N_heads]
        
        elif self.mode == 'infer':
            e = x[0]                            # e: [B, m, c], B=1
            a = -torch.ones((N_samples, N_steps)).long()       # a: {-1,0,1, ... N_logits-1} ^ [N_samples, N_steps]
            p = torch.ones((N_samples,)).float()
            
            for s in range(N_samples):
                for i in range(N_steps):
                    o, z = self.predict_action_logits(one_hot(a[s], num_classes=N_logits).float()[None], e)     #FIXME: How to represent the start action?
                    o, z = o[0], z[0]           # No use batch dim.  o: [N_steps, N_logits], z:[N_steps, N_features*N_heads]
                    prob = F.softmax(o[i], dim=0)
                    sampled_a = torch.multinomial(prob, 1)
                    _p = prob[sampled_a]
                    p[s] = p[s] * _p
                    a[s, i] = sampled_a
                    if i == 0:
                        z1 = z[0]             # [N_features*N_heads]
                    
            return a, p, z1                   # [N_samples, N_steps], [N_samples], [N_features*N_heads]
    
    
    def set_mode(self, mode):
        
        assert mode in ["train", "infer"]
        self.mode = mode
    
      
    def predict_action_logits(self,
                              a, e):
        
        N_steps = self.N_steps
        N_logits = self.N_logits
        N_features = self.N_features
        N_heads = self.N_heads
        N_layers = self.N_layers
        torso_feature_shape = self.torso_feature_shape
        
        batch_size = a.shape[0]
        
        x = self.linear_1(a.reshape(batch_size*N_steps, -1))           # [batch_size*N_steps, N_features*N_heads]
        x = self.pos_embed(torch.arange(0, N_steps).repeat(batch_size).float().view((-1,1))) + x    # [batch_size*N_steps, N_features*N_heads]
        x = x.reshape(batch_size, N_steps, -1)          # [batch_size, N_steps, N_features*N_heads]
        
        for layer in range(N_layers):
            x = self.self_layer_norms[layer](x)         # [batch_size, N_steps, N_features*N_heads]
            c = self.self_attentions[layer]([x,])       # [batch_size, N_steps, N_features*N_heads]
            if self.mode == 'train':
                c = self.self_dropouts[layer](c)
            x = x + c                                   # [batch_size, N_steps, N_features*N_heads]

            x = self.cross_layer_norms[layer](x)        # [batch_size, N_steps, N_features*N_heads]
            c = self.cross_attentions[layer]([x, e])    # [batch_size, N_steps, N_features*N_heads]
            if self.mode == 'train':
                c = self.cross_dropouts[layer](c)
            x = x + c                                   # [batch_size, N_steps, N_features*N_heads]
        
        o = self.linear_2(self.relu(x.reshape(-1, N_features*N_heads))).reshape(batch_size, N_steps, N_logits)  # [batch_size, N_steps, N_logits]
        
        return o, x           
    

class ValueHead(nn.Module):
    
    def __init__(self,
                 N_layers=3,
                 in_channel=2048,
                 inter_channel=512,
                 out_channel=8):
        
        super(ValueHead, self).__init__()
        self.out_channel = out_channel
        self.in_channel = in_channel
        self.inter_channel = inter_channel
        self.N_layers = N_layers
        self.mode = "train"
        
        self.in_linear = nn.Linear(in_channel, inter_channel)
        self.linaers = [nn.Linear(inter_channel, inter_channel) for _ in range(N_layers-1)]
        self.out_linear = nn.Linear(inter_channel, out_channel)
        self.relus = [nn.ReLU() for _ in range(N_layers)]
        
    
    def forward(self, x):
        
        # Input:
        #   [z]. z: The feature in policy head.
        
        N_layers = self.N_layers
        
        z = x[0]
        if self.mode == "infer":
            z = z[None]        
        z = self.relus[0](self.in_linear(z))
        for layer in range(N_layers-1):
            z = self.relus[layer+1](self.linaers[layer](z))
            
        q = self.out_linear(z)
        if self.mode == "infer":
            q = q[0]
        return q
    
    
    def set_mode(self, mode):
        
        assert mode in ["train", "infer"]
        self.mode = mode
        

class Net(nn.Module):
    '''
    网络部分
    '''
    
    def __init__(self,
                 T=7,
                 S_size=4,
                 N_steps=6,
                 coefficients=[0, 1, -1],
                 N_samples=32,
                 n_attentive=8,
                 N_heads=32,
                 N_features=64,
                 **kwargs):
        '''
        初始化部分
        '''
        
        super(Net, self).__init__()
        # Parameters.
        self.T = T
        self.S_size = S_size
        self.N_steps = N_steps
        self.coefficients = coefficients
        token_len = 3 * S_size // N_steps
        N_logits = len(coefficients) ** token_len   # len(F) ^ len(token)
        self.N_logits = N_logits
        self.token_len = token_len
        self.N_samples = N_samples
        
        # Network.
        self.torso = Torso(S_size=S_size, T=T,
                           n_attentive=n_attentive)
        self.policy_head = PolicyHead(N_steps=N_steps,
                                      N_logits=N_logits,
                                      N_samples=N_samples,
                                      N_heads=N_heads,
                                      N_features=N_features)
        self.value_head = ValueHead(in_channel=N_features*N_heads)
        self.mode = "train"
        
    
    def forward(self, x):
        
        # Input:
        #   If train mode:
        #     Tensors of shape of [B,T,S,S,S]. First one is current tensor. (numpy)
        #     Scalars of shape of [B,s].                                    (numpy)
        #     (Groundtruth) of shape of [B, N_steps]. 
        #   Elif infer mode:
        #     Tensors of shape of [T,S,S,S]
        #     Scalars of shape of [s,]             
        
        if self.mode == 'train':
            states, scalars, g = x
            self.policy_head.set_mode("train")
            self.value_head.set_mode("train")
            self.torso.set_mode("train")
            
            e = self.torso([states, scalars])
            o, z1 = self.policy_head([e, g])
            q = self.value_head([z1])
            
            return o, q          # o: [B, N_steps, N_logits]; q: [B, out_channels]
        
        elif self.mode == 'infer':
            states, scalars = x
            self.policy_head.set_mode("infer")
            self.value_head.set_mode("infer")
            self.torso.set_mode("infer")
            
            e = self.torso([states, scalars])
            a, p, z1 = self.policy_head([e])
            q = self.value_head([z1])
            
            return a, p, q        #FIXME: Neet to process q.
                                  # a: {0,1,..., N_logits-1} ^ [N_samples, N_steps]; p: [N_samples,]; q: [out_channels,]
        
    
    def set_mode(self, mode):
        
        assert mode in ["train", "infer"]
        self.mode = mode
        
    
    def logits_to_action(self, logits):
        '''
        logit: N_steps values of {0, 1, ..., N_logits - 1}.
        e.g.: 
            If:
                token_len = 2
                coefficients = [0, 1, -1]
                N_steps = 6 
            Then:    
                [0, 1, 2, 3, 4, 5] -> [0 0 | 0 1 | 0 -1 | 1 0 | 1 1 | 1 -1 ]
        '''
        token_len = self.token_len
        coefficients = self.coefficients
        action = []
        for logit in logits:                       # Get one action
            token = []
            for _ in range(token_len):             # Get one token
                idx = logit % len(coefficients)
                token.append(coefficients[idx])
                logit = logit // len(coefficients)
            token.reverse()
            action.extend(token)
        
        action = np.array(action, dtype=np.int32).reshape((3, -1))
        return action
        
        
    def action_to_logits(self,
                         action):
        '''
        action: A [3, S_size] array.
        '''
        
        # Break action into tokens.
        token_len = self.token_len
        coefficients = self.coefficients
        action = action.reshape((-1, token_len))     # [N_steps, token_len]
        
        # Get logits.
        logits = []
        for token in action:         # Get one logit.
            # token = token.to_list()
            logit = 0
            token = token[::-1]
            for idx, v in enumerate(token):
                logit += coefficients.index(v) * (len(coefficients) ** idx)
            logits.append(logit)
            
        return np.array(logits)
        
        
    def value(self, output, u_q=.75):
        '''
        根据网络的输出, 得到效用值
        output: output = net(x)
        '''
        q = output[-1]
        q = q.detach().cpu().numpy()
        out_channels = q.shape[0]
        
        j = math.ceil(u_q * out_channels)
        return q, q[(j-1):].mean()
    
    
    def policy(self, output):
        '''
        根据网络的输出, 得到采样的策略
        output: output = net(x)
        '''
        assert len(output) == 3, "We need the output from infer mode."

        a, p, _ = output
        a, p = a.detach().cpu().numpy(), p.detach().cpu().numpy()
        
        actions = []
        for logits in a:
            actions.append(self.logits_to_action(logits))
        actions = np.stack(actions, axis=0)
        
        return actions, p
        
    
    
if __name__ == '__main__':
    # torso = Torso()
    # test_input = [np.random.randint(-1, 1, (64, 7, 4, 4, 4)), np.random.randint(-1, 1, (64, 3))]
    # e = torso(test_input)
    # import pdb; pdb.set_trace()
    
    # policy_head = PolicyHead()
    # test_g = torch.tensor([0,1,2,3,4,5]).repeat(64).reshape(64, 6)
    # train_output = policy_head([e, test_g])
    # import pdb; pdb.set_trace()
    
    # value_head = ValueHead()
    # value_output = value_head([train_output[1]])
    # import pdb; pdb.set_trace()
    
    net = Net()
    test_input = [np.random.randint(-1, 1, (64, 7, 4, 4, 4)), np.random.randint(-1, 1, (64, 3))]
    test_g = torch.tensor([0,1,2,3,4,5]).repeat(64).reshape(64, 6)
    train_output = net([*test_input, test_g])  
    import pdb; pdb.set_trace()
    
    net.set_mode("infer")
    test_input = [np.random.randint(-1, 1, (7, 4, 4, 4)), np.random.randint(-1, 1, (3))]
    infer_output = net(test_input)
    import pdb; pdb.set_trace()
    
    _, value = net.value(infer_output)
    policy = net.policy(infer_output)
    import pdb; pdb.set_trace()
    
    # net = Net()
    # logits = np.array([0,1,2,2,1,1])
    # action = net.logits_to_action(logits)
    # logits = net.action_to_logits(action)
    # import pdb; pdb.set_trace()