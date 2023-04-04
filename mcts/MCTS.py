import numpy as np
import torch
import math
from tqdm import tqdm

import sys
import os
import copy
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join(".")))
from env import Environment
from net import Net
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
               net: Net):
        '''
        Expand this node.
        Return the value of this state.
        '''
        if not self.is_leaf:
            raise Exception("This node has been expanded.")
        self.is_leaf = False
        
        #FIXME: Here we can apply a transposition table.
        
        # Get state for net evaluation.
        # State: 
        #   Tensors: [cur_state, last t=1 action, last t=2 action, ... last t=T-1 action]
        #   Scalars: [depth(step_ct)]
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
        _, value, policy, prob = *net.value(output), *net.policy(output)    # policy: [N_samples, 3, S_size]
        policy = [canonicalize_action(action) for action in policy]
        
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
        node = self
        while node.parent is not None:
            action_idx = node.pre_action_idx
            node = node.parent
            node.N[action_idx] += 1
            node.Q[action_idx] += (value + -1 * (node.depth + 1))
    
    
    def select(self, c=.5):
        '''
        Choose the best child.
        Return the chosen node.
        '''
        if self.is_leaf:
            raise Exception("Cannot choose a leaf node.")
        
        scores = [self.Q[i] + c * self.pi[i] * math.sqrt(sum(self.N)) / (1 + self.N[i])
                  for i in range(self.children_n)]
        
        return self.children[np.argmax(scores)]
        
    

class MCTS():
    '''
    蒙特卡洛树搜索
    '''
    
    def __init__(self,
                 init_state,
                 simulate_times=400,
                 **kwargs):
        '''
        超参数传递
        '''
        
        self.simulate_times = simulate_times
        if init_state is not None:
            self.root_node = Node(state=init_state,
                                parent=None,
                                pre_action=None,
                                pre_action_idx=None)

    
    def __call__(self,
                 state,
                 net: Net):
        '''
        进行一次MCTS
        返回: action, actions, visit_pi
        '''

        assert is_equal(state, self.root_node.state), "State is inconsistent."

        for simu in tqdm(range(self.simulate_times)):
            # Select a leaf node.
            node = self.root_node
            while not node.is_leaf:
                node = node.select()         #FIXME: Need to control the factor c.
            node.expand(net)
        
        actions = self.root_node.actions
        N = self.root_node.N
        visit_ratio = (np.array(N) / sum(N)).tolist()
        action = actions[np.argmax(visit_ratio)]
        
        return action, actions, visit_ratio
        
    
    def move(self,
             action):
        '''
        MCTS向前一步
        '''
        assert not self.root_node.is_leaf, "Cannot move a leaf node."
        
        # Get the action idx.
        action_idx = None
        for idx, child_action in enumerate(self.root_node.actions):
            if is_equal(child_action, action):
                action_idx = idx
                
        self.root_node.children.append(copy.deepcopy(self.root_node.children[action_idx]))
        self.root_node.children.reverse()
        [self.root_node.children.pop() for _ in range(idx+1)]
        self.root_node = self.root_node.children[0]
        
        
    def reset(self,
              state,
              simulate_times=None):
        '''
        Reset MCTS.
        '''
        if simulate_times is not None:
            self.simulate_times = simulate_times
        self.root_node = Node(state=state,
                            parent=None,
                            pre_action=None,
                            pre_action_idx=None)
    
    
    
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
                     pre_action=None,
                     pre_action_idx=None)
    
    from net import Net
    net = Net(N_samples=4)   # For debugging.
    
    ############ Debug for Node ############
    # import pdb; pdb.set_trace()
    # root_node.expand(net)
    # children_node = root_node.select()
    # children_node.expand(net)
    # import pdb; pdb.set_trace()
    
    ############ Debug for MCYS ############
    mcts = MCTS(init_state=init_state,
                simulate_times=20)
    import pdb; pdb.set_trace()
    action, actions, pi = mcts(init_state, net)
    import pdb; pdb.set_trace()
    mcts.move(action)
    state = init_state - action2tensor(action)
    import pdb; pdb.set_trace()
    action, actions, pi = mcts(state, net)
    import pdb; pdb.set_trace()