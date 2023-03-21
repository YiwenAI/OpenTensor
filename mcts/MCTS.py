from ..env import Environment

class Node():
    '''
    一个MCTS的节点
    '''
    
    def __init__(self):
        #####
        # 这里应该初始化一个节点，
        # 包括Q, N以及女儿父母
        # e.g.:
        # self.Q = 0
        # ......
        #####
        pass
    
# todo: 采样部分
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