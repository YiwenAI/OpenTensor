import numpy as np
from math import sqrt


# todo: 建议写成gym的标准形式 init reset step
# todo: 可以少写函数 多写变量

def outer(x, y, z):
    # 得到三维张量，三维分别表示xyz
    return np.einsum('i,j,k->ijk', x, y, z)

class Environment():
    '''
    负责定义游戏的动作、状态以及回报
    state: np.darray，[4, 4, 4]
    action: np.darray，[3, 4]  (表示u, v, w)
    results: [[s_1, a_1, pi_1, r_1], ...]
    
    包括的功能有:
        play: 进行一次游戏
        生成人工Tensor
        以及其它和state, action及reward相关的操作
    '''

    def __init__(self,
                 S_size,
                 R_limit,
                 actions=None,
                 init_state=None,
                 **kwargs):
        '''
        S_size: u, v, w的维度
        R_limit: 游戏的步数上限
        '''
        self.S_size = S_size
        self.R_limit = R_limit
        if init_state is None:
            init_state = self.get_init_state(S_size)
        self.cur_state = init_state
        pass
    
    def get_init_state(self,
                       S_size):
        '''
        得到一个初始化状态: state
        S_size: u, v, w的维度
        返回: state.
        '''
        #####
        # 注意，这里我们可以添加基变换的数据增强
        #####
        
        def one_hot(idx):
            temp = np.zeros((S_size, ), dtype=np.uint8)
            temp[idx] = 1
            return temp
        
        # 1. Get the original Matmul-Tensor.
        init_state = np.zeros((S_size, S_size, S_size), dtype=np.uint8)
        n = round(sqrt(S_size))
        
        for i in range(n):                  # 用自然基的方式构建向量
            for j in range(n):
                z_idx = i * n + j
                z = one_hot(z_idx)          # C_{i,j} = c_{i*n + j}
                for k in range(n):
                    x_idx = i * n + k       # A_{i,k} = a_{i*n + k}
                    y_idx = k * n + j       # B_{k,j} = b_{k*n + j}
                    x, y = one_hot(x_idx), one_hot(y_idx)
                    init_state += outer(x, y, z)
        
        # 2. Change of Basis.
        # raise NotImplementedError
        return init_state
            
    def step(self,
             action):
        '''
        状态转移并返回此次操作的奖励。
        返回: reward
        '''
        u, v, w = action
        self.cur_state -= outer(u, v, w)
        return -1
    
    def terminate_reward(self):
        '''
        截断时得到的惩罚。
        返回: reward
        '''
        state = self.cur_state
        terminate_reward = 0
        for z_idx in range(self.S_size):
            terminate_reward -= np.linalg.matrix_rank(np.mat(state[..., z_idx], dtype=np.uint8))
        return terminate_reward
    
    def is_terminate(self):
        '''
        判断cur_state是否为0
        返回: bool
        '''
        return np.all(self.cur_state == 0)
    
    def generate_synthetic_examples(self,
                                    samples_n):
        '''
        生成人工合成的Tensor examples
        返回: results
        '''
        pass
    
    def reset(self,
              init_state=None):
        '''
        重置环境
        '''
        self.cur_state = init_state
        
        
if __name__ == '__main__':
    test_env = Environment(S_size=4,
                           R_limit=128)
    import pdb; pdb.set_trace()