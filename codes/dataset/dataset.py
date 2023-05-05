import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

import sys
import os
import copy
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join(".")))
from codes.utils import *

class TupleDataset(Dataset):
    
    def __init__(self,
                 T,
                 S_size,
                 N_steps,
                 coefficients,
                 self_data=[],
                 synthetic_data=[],
                 debug=False,
                 save_type="traj"):
        # examples: A list of episodes, including:
        #   1. state (Network input)
        #       1.1 tensors (np)
        #       1.2 scalars (np)
        #   2. action (np)
        #   3. reward
        
        self.T = T
        self.S_size = S_size
        self.N_steps = N_steps
        self.coefficients = coefficients
        token_len = 3 * S_size // N_steps
        self.N_logits = len(coefficients) ** token_len
        self.ct = 0
        
        print("Preprocessing dataset...")
        self.self_data = self_data
        self.synthetic_data = synthetic_data
        self.self_examples = []
        self.synthetic_examples = []
            
        #TODO: Randomize sign permutation.
        #TODO: Reformualte data format.            
        # Canonicalize actions & to logits.
        if save_type == "tuple":
            for episode in tqdm(synthetic_data):
                state, action, reward = episode
                action = self.action_to_logits(canonicalize_action(action))
                self.synthetic_examples.append([state, action, reward])
            for episode in tqdm(self_data):
                state, action, reward = episode
                action = self.action_to_logits(canonicalize_action(action))
                self.self_examples.append([state, action, reward])
            self.examples = self.self_examples + self.synthetic_examples
                
        else:   # Traj format data.
            self.prepare_examples_from_trajs()
                   

    def prepare_examples_from_trajs(self):
        '''
        This function will permutate self.xxx_data (but not change),
        and get the corresponding examples.
        '''
        S_size = self.S_size
        T = self.T
        
        def traj_to_episode(traj):
            results = []
            states, actions, rewards = traj
            states.reverse(); actions.reverse(); rewards.reverse()
            actions_tensor = [action2tensor(action) for action in actions]
            for idx, state in enumerate(states):
                tensors = np.zeros((T, S_size, S_size, S_size), dtype=np.int32)
                tensors[0] = state            # state.
                if idx != 0:
                    # History actions.
                    tensors[1:(idx+1)] = np.stack(reversed(actions_tensor[max(idx-(T-1), 0):idx]), axis=0)        
                scalars = np.array([idx, idx, idx])     #FIXME: Havn't decided the scalars.
                
                cur_state = [tensors, scalars]
                action = self.action_to_logits(canonicalize_action(actions[idx]))
                reward = rewards[idx]
                results.append([cur_state, action, reward])
            return results
        
        
        self_examples, synthetic_examples = [], []
        
        for traj in self.synthetic_data:
            _, actions, rewards = traj     # [T, S, S, S], [T, 3, S], [T]
            # Shuffle the traj.
            np.random.shuffle(actions)
            new_states = []
            sample = np.zeros((S_size, S_size, S_size), dtype=np.int32)
            for idx, action in enumerate(actions):
                sample = sample + action2tensor(action)
                new_states.append(sample.copy())
            new_traj = [new_states, actions, rewards]
            synthetic_examples.extend(traj_to_episode(new_traj))
            
        for traj in self.self_data:
            _, actions, rewards = traj
            np.random.shuffle(actions)
            new_states = []
            sample = np.zeros((S_size, S_size, S_size), dtype=np.int32)
            for idx, action in enumerate(actions):
                sample = sample + action2tensor(action)
                new_states.append(sample.copy())
            new_traj = [new_states, actions, rewards]
            self_examples.extend(traj_to_episode(new_traj))                
            
        self.self_examples, self.synthetic_examples = self_examples, synthetic_examples
        self.examples = self_examples + synthetic_examples
        
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):        
        state, action, reward = self.examples[idx]
        tensor, scalar = state
        action = self.logits_to_action(action)
        tensor, action = self.random_sign_permutation(tensor, action)
        action = canonicalize_action(action)           #FIXME: Is it needed?
        action = self.action_to_logits(action)
        return [tensor, scalar], action, reward
    
    def action_to_logits(self,
                         action):
        '''
        action: A [3, S_size] array.
        '''
        
        # Break action into tokens.
        token_len = 3 * self.S_size // self.N_steps
        coefficients = self.coefficients
        action = action.reshape((-1, token_len))     # [N_steps, token_len]
        
        # Get logits.
        logits = []                  # Start sign.
        for token in action:         # Get one logit.
            # token = token.to_list()
            logit = 0
            if torch.is_tensor(token):
                token = torch.flip(token, dims=(0,))
            else:
                token = token[::-1]
            for idx, v in enumerate(token):
                logit += coefficients.index(v) * (len(coefficients) ** idx)
            logits.append(logit)
            
        return np.array(logits, dtype=np.int32)
    
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
        token_len = 3 * self.S_size // self.N_steps
        coefficients = self.coefficients
        action = []
        for logit in logits:                       # Get one action
            token = []
            if logit == self.N_logits:
                raise                              # Mean that there is a start sign in the middle of action.            
            for _ in range(token_len):             # Get one token
                idx = logit % len(coefficients)
                token.append(coefficients[idx])
                logit = logit // len(coefficients)
            token.reverse()
            action.extend(token)
        
        action = np.array(action, dtype=np.int32).reshape((3, -1))
        return action    
    
    def random_sign_permutation(self,
                                tensor, action):
        trans_1, trans_2, trans_3 = \
            (np.random.binomial(1, .5, self.S_size) * 2 - 1).astype(np.int32), \
            (np.random.binomial(1, .5, self.S_size) * 2 - 1).astype(np.int32), \
            (np.random.binomial(1, .5, self.S_size) * 2 - 1).astype(np.int32)
        tensor = np.einsum('i, j, k, bijk -> bijk', trans_1, trans_2, trans_3, tensor,
                           dtype=np.int32)
        action = np.stack([action[0]*trans_1, action[1]*trans_2, action[2]*trans_3], axis=0)
        return tensor, action
    
    
if __name__ == '__main__':
    dataset = TupleDataset(S_size=4,
                           N_steps=6,
                           coefficients=[0, 1, -1],
                           synthetic_examples=np.load("data/100000_T5_scalar3.npy", allow_pickle=True).tolist(),
                           debug=True)
    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # res = next(iter(dataloader))
    import pdb; pdb.set_trace()