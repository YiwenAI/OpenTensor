import numpy as np
import torch
import math

import sys
import os
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join(".")))
from env import Environment
from utils import *


class Node():
    '''
    一个MCTS的节点
    '''
    
    def __init__(self,
                 state,
                 parent,
                 pre_action,
                 pre_action_idx):
        #####
        # 这里应该初始化一个节点，
        # 包括Q, N以及女儿父母
        # e.g.:
        # self.Q = 0
        # ......
        #####
        
        self.parent = parent    # parent: A Node instance (or None).
        self.pre_action = pre_action     # pre_action: Action (or None).
        self.pre_action_idx = pre_action_idx
        self.is_leaf = True
        self.state = state      # state: Tensor.
        
        self.actions = []       # A list for actions.
        self.children = []      # A list for nodes.
        self.N = []             # A list for visit counts.
        self.Q = []             # A list for action value.
        self.pi = []            # A list for empirical policy probability.
        self.children_n = 0
        
        node = self
        depth = 0
        while node.parent:
            depth += 1 
            node = node.parent
        self.depth = depth
        
        
    def expand(self,
               net):
        '''
        Expand this node.
        Return the value of this state.
        '''
        if not self.is_leaf:
            raise Exception("This node has been expanded.")
        self.is_leaf = False
        
        #FIXME: Here we can apply a transposition table.
        
        # Get state for net evaluation.
        T = net.T
        tensors = np.zeros([T, *self.state.shape], dtype=np.int32)
        tensors[0] = self.state       # Current state.
        node = self
        for t in range(1, T):
            if node.parent is None:
                break
            tensors[t] = action2tensor(node.pre_action)
            node = node.parent
        scalars = np.array([self.depth, self.depth, self.depth]) #FIXME: Havn't decided the scalars.
        
        net.set_mode("infer")
        output = net([tensors, scalars])
        value, policy, prob = net.value(output), *net.policy(output)    # policy: [N_samples, 3, S_size]
        
        # Get empirical policy probability.
        N_samples = net.N_samples
        rec = [False for _ in range(N_samples)]    # "True" represents having been recorded.
        actions = []
        pi = []
        for pos in range(N_samples):               # Naive loop.
            action = policy[pos]
            if not rec[pos]:
                # Count.
                actions.append(action)
                rec[pos] = True
                ct = 1
                for i in range(pos+1, N_samples):
                    if rec[i]:                   # Have been counted.
                        continue 
                    if is_equal(policy[i], action):
                        ct += 1
                        rec[i] = True
                pi.append(ct / N_samples)
                
        self.actions = actions
        self.pi = pi
        self.children_n = len(actions)
              
        # Init records.
        self.N = [0 for _ in range(len(actions))]
        self.Q = [0 for _ in range(len(actions))]
        
        # Expand the children nodes.
        for idx, action in enumerate(actions):
            child_state = self.state - action2tensor(action)
            child_node = Node(state=child_state,
                              parent=self,
                              pre_action=action,
                              pre_action_idx=idx)
            self.children.append(child_node)
            
        # Backward propagate.
        ####### We are here! #######
    
    
    def select(self, c=.5):
        '''
        Choose the best child.
        Return the chosen node.
        '''
        if not self.is_leaf:
            raise Exception("Cannot choose a leaf node.")
        
        scores = [self.Q[i] + c * self.pi[i] * math.sqrt(sum(self.N)) / (1 + self.N[i])
                  for i in range(self.children_n)]
        
        return self.children[np.argmax(scores)]
    

class MCTS():
    '''
    蒙特卡洛树搜索
    '''
    
    def __init__(self,
                 **kwargs):
        '''
        超参数传递
        '''
        # self.simu_nums = simu_nums
        # ......
        self.env = Environment()
        self._init_tree()
        pass
    
    def __call__(self,
                 state,
                 net):
        '''
        进行一次MCTS
        返回: action, actions, pi
        '''
        # 初始化
        self._init_tree(state)
        #####
        # MCTS搜索部分
        # while True:
        #     self._expand()
        #     self._choose()
        #     self._simulate(net)
        #     self._propagate()
        #####
        
        # 搜索完毕后返回结果
        return self.get_results()
        
    def set_env(self, env):
        '''
        传入环境
        '''
        self.env = env
        
    def _init_tree(self):
        '''
        重新初始化一棵树
        树用成员变量来表示
        操作单位为Node
        '''
        pass
    
    def _expand(self):
        pass
    
    def _choose(self):
        pass
    
    def _simulate(self,
                  net):
        '''
        神经网络指导模拟
        '''
        pass
    
    def _propagate(self):
        pass
    
    def get_results(self):
        '''
        返回MCTS的结果
        返回: action, actions, pi
        '''
        pass
    
    def copy(self):
        '''
        复制MCTS
        '''
        pass
    
    def move(self,
             action):
        '''
        MCTS向前一步
        '''
        pass
    
    
    
if __name__ == '__main__':
    
    init_state = np.array(
        [[[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]],

       [[0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0]],

       [[0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]],

       [[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]])
    root_node = Node(state=init_state,
                     parent=None,
                     pre_action=None)
    
    from net import Net
    net = Net(N_samples=4)   # For debugging.
    
    import pdb; pdb.set_trace()
    root_node.expand(net)
    children_node = root_node.children[0]
    import pdb; pdb.set_trace()