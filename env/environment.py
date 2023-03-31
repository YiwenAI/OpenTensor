import numpy as np
from math import sqrt

import sys
import os
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join(".")))
from utils import *

# todo: 建议写成gym的标准形式 init reset step
# todo: 可以少写函数 多写变量

class Environment():
    '''
    负责定义游戏的动作、状态以及回报
    state: np.darray, [4, 4, 4]
    action: np.darray, [3, 4]  (表示u, v, w)
    results: [[s_1, a_1, pi_1, r_1], ...]
    
    包括的功能有:
        play: 进行一次游戏
        生成人工Tensor
        以及其它和state, action及reward相关的操作
    '''

    def __init__(self,
                 S_size,
                 R_limit,
                 init_state=None,
                 T=7,
                 **kwargs):
        '''
        S_size: u, v, w的维度
        R_limit: 游戏的步数上限
        '''
        # 参数
        self.S_size = S_size
        self.R_limit = R_limit
        self.T = T
        # 环境变量
        if init_state is None:
            init_state = self.get_init_state(S_size)
        self.cur_state = init_state
        self.accumulate_reward = 0
        self.step_ct = 0
        
        # 历史变量
        self.hist_actions = [np.zeros_like(self.cur_state) for _ in range(self.T-1)]
        
    
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
            temp = np.zeros((S_size, ), dtype=np.int32)
            temp[idx] = 1
            return temp
        
        # 1. Get the original Matmul-Tensor.
        init_state = np.zeros((S_size, S_size, S_size), dtype=np.int32)
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
        #FIXME: We haven't applied "basis change" operation.
        # raise NotImplementedError
        return init_state
            
    def step(self,
             action):
        '''
        状态转移并改动reward, 并返回是否游戏结束
        '''
        u, v, w = action
        self.cur_state -= outer(u, v, w)
        self.accumulate_reward -= 1
        self.step_ct += 1
        self.hist_actions.append(action2tensor(action))
        # 判断是否终止
        if self.is_terminate():
            return True
        if self.step_ct >= self.R_limit:
            self.accumulate_reward += self.terminate_reward()
            return True
        return False
    
    def terminate_reward(self):
        '''
        截断时得到的惩罚。
        返回: reward
        '''
        state = self.cur_state
        terminate_reward = 0
        for z_idx in range(self.S_size):
            terminate_reward -= np.linalg.matrix_rank(np.mat(state[..., z_idx], dtype=np.int32))
        return terminate_reward
    
    def is_terminate(self):
        '''
        判断cur_state是否为0
        返回: bool
        '''
        return is_zero_tensor(self.cur_state)
    
    def reset(self,
              init_state=None):
        '''
        重置环境
        '''
        if init_state is None:
            init_state = self.get_init_state(self.S_size)        
        self.cur_state = init_state
        self.accumulate_reward = 0
        self.step_ct = 0
        self.hist_actions = [np.zeros_like(self.cur_state) for _ in range(self.T-1)]
        
    def get_network_input(self):
        '''
        将变量组织成网络输入的格式
        '''
        T = self.T
        S_size = self.S_size
        hist_actions = self.hist_actions[-(T-1):]
        hist_actions.reverse()
        tensors = np.zeros((T, S_size, S_size, S_size), dtype=np.int32)
        tensors[0] = self.cur_state
        tensors[1:] = np.stack(hist_actions, axis=0)
        scalars = np.array([self.step_ct, self.step_ct, self.step_ct])  #FIXME: Havn't decided the scalars.
        
        return tensors, scalars
        
        
if __name__ == '__main__':
    test_env = Environment(S_size=4,
                           R_limit=8)
    test_action = np.array([
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0]
    ])
    for _ in range(8):
        print(test_env.step(test_action))
        print(test_env.accumulate_reward)
    import pdb; pdb.set_trace()