import numpy as np

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