import torch
import numpy as np
from torch.utils.data import Dataset

class TupleDataset(Dataset):
    
    def __init__(self, file_path=None, examples=None):
        # examples: A list of episodes, including:
        #   1. state (Network input)
        #       1.1 tensors (np)
        #       1.2 scalars (np)
        #   2. action (np)
        #   3. reward
        if examples is not None:
            self.examples = examples
        else:
            self.examples = np.load(file_path, allow_pickle=True)
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        state, action, reward = self.examples[idx]
        tensor, scalar = state
        return [tensor, scalar], action, reward
    
    
if __name__ == '__main__':
    dataset = TupleDataset("./data/100000_T5_scalar3.npy")
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    res = next(iter(dataloader))
    import pdb; pdb.set_trace()