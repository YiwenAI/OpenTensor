import os
from tqdm import tqdm
import numpy as np

S_size = 4

data_name = "100000_S4T7_scalar3.npy"
data_path = os.path.join(".", "data", "traj_data", data_name)

data = np.load(data_path, allow_pickle=True)
filtered_data = []; ct = 0

for traj_idx, traj in tqdm(enumerate(data)):
    _, actions, _ = traj
    raw_r = len(actions)
    flag = False
    for (i, j) in [[0,1], [1,2], [2,0]]:
        _mat = np.zeros((S_size ** 2, raw_r), dtype=np.int32)
        for idx, action in enumerate(actions):
            _mat[:, idx] = np.outer(action[i], action[j]).reshape((-1,))
        if np.linalg.matrix_rank(_mat) < raw_r:
            ct += 1
            print("Found redundancy...")
            flag = True
            break
        
    if not flag:
        filtered_data.append(traj)
        
import pdb; pdb.set_trace()