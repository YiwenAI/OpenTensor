import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
            mask = torch.from_numpy(np.triu(np.ones([N_heads, N_x, N_y]), k=1)).float()
            scores = scores * mask
            
        o_s = torch.matmul(scores, v_s)   # [N_heads, N_x, d]
        o_s = o_s.transpose(0, 1).contiguous().view(-1, N_heads*d)    # [N_x, N_heads*d]
        x = x + self.linear_1(o_s)                                    # [N_x, c_x]
        
        x = x + self.linear_3(self.gelu(self.linear_2(self.final_layer_norm(x))))     # [N_x, c_x]
        
        return x
        

class PolicyHead(nn.Module):

    def __init__(self,
                 N_steps=6,        #FIXME
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
        self.self_layer_norms = [nn.LayerNorm((N_steps, N_features * N_heads)) for _ in range(N_layers)]
        self.cross_layer_norms = [nn.LayerNorm((N_steps, N_features * N_heads)) for _ in range(N_layers)]
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
        #   e: [m, c]
        #   g: {0,1,... N_logits-1} ^ N_steps, [N_steps]
        
        N_steps = self.N_steps
        N_logits = self.N_logits
        N_samples = self.N_samples
        assert self.mode in ['train', 'infer']
        
        if self.mode == 'train':
            e, g = x                            # g: {0,1,... N_logits-1} ^ N_steps, [N_steps]
            g_onehot = F.one_hot(g, num_classes=N_logits).float()     # [N_steps, N_logits]
            #FIXME: We haven't applied "shift" operation.
            o, z = self.predict_action_logits(g_onehot, e)    # o: [N_steps, N_logits]; z: [N_steps, N_features*N_heads]
            return o, z[0]
        
        elif self.mode == 'infer':
            e = x[0]
            a = torch.zeros((N_samples, N_steps)).long()       # a: {0,1, ... N_logits-1} ^ [N_samples, N_steps]
            p = torch.ones((N_samples,)).float()
            
            for s in range(N_samples):
                for i in range(N_steps):
                    o, z = self.predict_action_logits(F.one_hot(a[s], num_classes=N_logits).float(), e)     #FIXME: How to represent the start action?
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
        
        x = self.linear_1(a)           # [N_steps, N_features*N_heads]
        x = self.pos_embed(torch.arange(0, N_steps).float().view((-1,1))) + x    # [N_steps, N_features*N_heads]
        
        for layer in range(N_layers):
            x = self.self_layer_norms[layer](x)         # [N_steps, N_features*N_heads]
            c = self.self_attentions[layer]([x,])       # [N_steps, N_features*N_heads]
            if self.mode == 'train':
                c = self.self_dropouts[layer](c)
            x = x + c                                   # [N_steps, N_features*N_heads]
            x = self.cross_layer_norms[layer](x)        # [N_steps, N_features*N_heads]
            c = self.cross_attentions[layer]([x, e])    # [N_steps, N_features*N_heads]
            if self.mode == 'train':
                c = self.cross_dropouts[layer](c)
            x = x + c                                   # [N_steps, N_features*N_heads]
        
        o = self.linear_2(self.relu(x))                 # [N_steps, N_logits]
        
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
        
        self.in_linear = nn.Linear(in_channel, inter_channel)
        self.linaers = [nn.Linear(inter_channel, inter_channel) for _ in range(N_layers-1)]
        self.out_linear = nn.Linear(inter_channel, out_channel)
        self.relus = [nn.ReLU() for _ in range(N_layers)]
        
    
    def forward(self, x):
        
        # Input:
        #   [z]. z: The feature in policy head.
        
        N_layers = self.N_layers
        
        z = x[0]
        z = self.relus[0](self.in_linear(z))[None]
        for layer in range(N_layers-1):
            z = self.relus[layer+1](self.linaers[layer](z))
            
        q = self.out_linear(z)
        return q[0]
        

class Net(nn.Module):
    '''
    网络部分
    '''
    
    def __init__(self,
                 **kwargs):
        '''
        初始化部分
        '''
        
        super(Net, self).__init__()
        self.torso = Torso()
        self.policy_head = PolicyHead()
        self.value_head = ValueHead()
        self.mode = "train"
        
    
    def forward(self, x):
        
        # Input:
        #   Tensors of shape of [T,S,S,S]. First one is current tensor. (numpy)
        #   Scalars of shape of [s,].                                   (numpy)
        #   (Groundtruth) of shape of [N_steps, N_logits].              
        
        if self.mode == 'train':
            states, scalars, g = x
            self.policy_head.set_mode("train")
            
            e = self.torso([states, scalars])
            o, z1 = self.policy_head([e, g])
            q = self.value_head([z1])
            
            return o, q
        
        elif self.mode == 'infer':
            states, scalars = x
            self.policy_head.set_mode("infer")
            
            e = self.torso([states, scalars])
            a, p, z1 = self.policy_head([e])
            q = self.value_head([z1])
            
            return a, p, q        #FIXME: Neet to process p.
        
    
    def set_mode(self, mode):
        
        assert mode in ["train", "infer"]
        self.mode = mode
    
    
    
if __name__ == '__main__':
    # torso = Torso()
    # test_input = [np.random.randint(-1, 1, (7, 4, 4, 4)), np.random.randint(-1, 1, (3,))]
    # e = torso(test_input)
    
    # policy_head = PolicyHead()
    # test_g = torch.tensor([0,1,2,3,4,5])
    # train_output = policy_head([e, test_g])
    # import pdb; pdb.set_trace()
    
    # value_head = ValueHead()
    # value_output = value_head([train_output[1]])
    # import pdb; pdb.set_trace()
    
    # policy_head.set_mode("infer")
    # infer_output = policy_head([e])
    # import pdb; pdb.set_trace()
    
    net = Net()
    test_input = [np.random.randint(-1, 1, (7, 4, 4, 4)), np.random.randint(-1, 1, (3,))]
    test_g = torch.tensor([0,1,2,3,4,5])
    train_output = net([*test_input, test_g])
    import pdb; pdb.set_trace()
    
    net.set_mode("infer")
    infer_output = net(test_input)
    import pdb; pdb.set_trace()