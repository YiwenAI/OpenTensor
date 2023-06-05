import numpy as np
import torch
import math
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
import os
import copy
from typing import Tuple
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join(".")))
from codes.env import Environment
from codes.net import Net
from codes.utils import *


class Node():
    '''
    一个MCTS的节点
    '''
    
    def __init__(self,
                 state,
                 parent,
                 pre_action,
                 pre_action_idx,
                 is_terminal=False):
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
        self.is_terminal = is_terminal
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
               net: Net,
               noise=False,
               network_output=None,
               R_limit=12):
        '''
        Expand this node.
        Return the value of this state.
        '''
        # 1. Check terminal situation.
        if not self.is_leaf:
            raise Exception("This node has been expanded.")
        self.is_leaf = False
        
        if self.is_terminal:    # Mean the state is terminal. Only propagate.
            node = self
            node.is_leaf = True
            if is_zero_tensor(node.state):
                value = 0
            else:
                value = -1 * terminate_rank_approx(node.state)
            while node.parent is not None:
                action_idx = node.pre_action_idx
                node = node.parent
                node.N[action_idx] += 1
                v = (value + -1 * (self.depth - node.depth))
                node.Q[action_idx] = v / node.N[action_idx] +\
                                    node.Q[action_idx] * (node.N[action_idx] - 1) / node.N[action_idx]   
            
            return         
        
        #FIXME: Here we can apply a transposition table.
        # 2. Get network output.
        # 2.1. If use network to infer:
        if network_output is None:
            tensors, scalars = self.get_network_input(net)
            
            net.set_mode("infer")
            with torch.no_grad():
                output = net([tensors, scalars])
                _, value, policy, prob = *net.value(output), *net.policy(output)    # policy: [1, N_samples, 3, S_size]
                del output, tensors, scalars
                value, policy = value[0], policy[0]
            policy = [canonicalize_action(action) for action in policy]
        
        # 2.2. If we already have network output:
        else:
            value, policy = network_output
        
        # 2.3. Add noise for root node's expand.
        if noise:
            noise_actions = [canonicalize_action(random_action()) for _ in range(len(policy) // 4)]
            policy = policy + noise_actions
        
        # 3. Get empirical policy probability.
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
        # import pdb; pdb.set_trace()
        self.actions = actions
        self.pi = pi
        self.children_n = len(actions)
              
        # 4. Init records.
        self.N = [0 for _ in range(len(actions))]
        self.Q = [0 for _ in range(len(actions))]
        
        # 5. Expand the children nodes.
        for idx, action in enumerate(actions):
            child_state = self.state - action2tensor(action)
            child_depth = self.depth + 1
            child_node = Node(state=child_state,
                              parent=self,
                              pre_action=action,
                              pre_action_idx=idx,
                              is_terminal=(is_zero_tensor(child_state) or child_depth >= R_limit))
            self.children.append(child_node)
            
        # 6. Backward propagate.
        node = self
        while node.parent is not None:
            action_idx = node.pre_action_idx
            node = node.parent
            node.N[action_idx] += 1
            v = (value + -1 * (self.depth - node.depth))
            node.Q[action_idx] = v / node.N[action_idx] +\
                                 node.Q[action_idx] * (node.N[action_idx] - 1) / node.N[action_idx]
    
    
    def select(self, c=None):
        '''
        Choose the best child.
        Return the chosen node.
        '''
        if self.is_leaf:
            raise Exception("Cannot choose a leaf node.")
        
        if c is None:
            c = 1.25 + math.log((1+19652+sum(self.N)) / 19652)
        
        scores = [self.Q[i] + c * self.pi[i] * math.sqrt(sum(self.N)) / (1 + self.N[i])
                  for i in range(self.children_n)]
        
        return self.children[np.argmax(scores)], scores
    
    
    def get_network_input(self, net):
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
        
        return tensors, scalars
        
    

class MCTS():
    '''
    蒙特卡洛树搜索
    '''
    
    def __init__(self,
                 init_state,
                 simulate_times=400,
                 R_limit=12,
                 **kwargs):
        '''
        超参数传递
        '''
        
        self.simulate_times = simulate_times
        self.R_limit = R_limit
        if init_state is not None:
            self.root_node = Node(state=init_state,
                                  parent=None,
                                  pre_action=None,
                                  pre_action_idx=None)

    
    def __call__(self,
                 state,
                 net: Net,
                 log=False,
                 verbose=False,
                 noise=False):
        '''
        进行一次MCTS
        返回: action, actions, visit_pi
        '''

        assert is_equal(state, self.root_node.state), "State is inconsistent."
        iter_item = range(self.simulate_times) if verbose else tqdm(range(self.simulate_times))
        R_limit = self.R_limit
        for simu in iter_item:
            # Select a leaf node.
            node = self.root_node
            while not node.is_leaf:
                node, scores = node.select()         #FIXME: Need to control the factor c.
            node.expand(net, noise=noise, R_limit=R_limit)
        
        actions = self.root_node.actions
        N = self.root_node.N
        visit_ratio = (np.array(N) / sum(N)).tolist()
        action = actions[np.argmax(visit_ratio)]
        
        if log:
            log_txt = self.log()
            return action, actions, visit_ratio, log_txt
        
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
                
        # Delete other children and move.
        self.root_node.children = [self.root_node.children[action_idx]]
        self.root_node = self.root_node.children[0]
        
        
    def reset(self,
              state,
              simulate_times=None,
              R_limit=None):
        '''
        Reset MCTS.
        '''
        if simulate_times is not None:
            self.simulate_times = simulate_times
        if R_limit is not None:
            self.R_limit = R_limit
        self.root_node = Node(state=state,
                            parent=None,
                            pre_action=None,
                            pre_action_idx=None)
        
        
    def visualize(self):
        '''
        visualize the tree.
        '''
        # Create a graph.
        graph = nx.DiGraph()
        close_set = [self.root_node]
        
        while close_set != []:
            node = close_set.pop()
            if not node.is_leaf:
                [graph.add_edge(
                    node,
                    child
                ) for child in node.children]
                [close_set.append(child) for child in node.children]
                
        nx.draw(graph, with_labels=True, font_weight='bold')
        raise NotImplementedError
        plt.show()
    
    
    def log(self):
        '''
        Get the log text.
        '''
        node = self.root_node
        state_txt = str(node.state)    # state.
        _, scores = node.select()
        N, Q, scores, children = np.array(node.N), np.array(node.Q), np.array(scores), np.array(node.actions)
        top_k_idx = np.argsort(N)[-5:]
        N, Q, scores, children = N[top_k_idx], Q[top_k_idx], scores[top_k_idx], children[top_k_idx]
        
        N_txt, Q_txt, scores_txt, children_txt = str(N), str(Q), str(scores), str(children)
        
        log_txt = "\n".join(
            ["\nCur state: \n", state_txt,
             "\nDepth: \n", str(node.depth),
             "\nchildren: \n", children_txt,
            "\nscores: \n", scores_txt,
            "\nQ: \n", Q_txt,
            "\nN: \n", N_txt,]
        )     
        
        return log_txt   
    
    
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