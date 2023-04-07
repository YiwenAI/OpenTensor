import numpy as np
import torch

def numpy_cvt(a):
    if torch.is_tensor(a):
        return a.numpy()
    return a

def outer(x, y, z):
    # 得到三维张量，三维分别表示xyz
    return np.einsum('i,j,k->ijk', x, y, z, dtype=np.int32)

def action2tensor(action):
    u, v, w = action
    return np.einsum('i,j,k->ijk', u, v, w, dtype=np.int32)

def is_zero_tensor(tensor):
    return np.all(tensor == 0)

def is_equal(a, b):
    assert a.shape == b.shape
    return np.all((a - b) == 0)

def canonicalize_action(action):
    action = numpy_cvt(action)
    u, v, w = action
    for e in u:
        if e != 0:
            u = (u * ((e > 0) * 2 - 1)).astype(np.int32)
            break
    for e in v:
        if e != 0:
            v = (v * ((e > 0) * 2 - 1)).astype(np.int32)
            break
    return np.stack([u, v, w])

def one_hot(a_s, num_classes):
    if len(a_s.shape) == 1:
        result = torch.zeros((a_s.shape[0], num_classes)).long()
        for idx, a in enumerate(a_s):
            if a == -1:
                continue
            result[idx, a] = 1
        return result
    elif len(a_s.shape) == 2:
        result = torch.zeros((a_s.shape[0], a_s.shape[1], num_classes)).long()
        for batch, a_batch in enumerate(a_s):
            for idx, a in enumerate(a_batch):
                if a == -1:
                    continue
                result[batch, idx, a] = 1
        return result