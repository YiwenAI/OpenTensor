from ..mcts import MCTS

class AlphaTensor():
    '''
    智能体模块
    '''
    
    def __init__(self,
                 net,
                 **kwargs):
        '''
        初始化一个智能体
        net: 一个网络实例
        '''
        self.net = net
        self.mcts = MCTS()
        #####
        # 其它超参数传递省略
        # e.g.:
        # self.simus_n = simus_n
        # ......
        #####
        
    def policy(self,
               state,
               **kwargs):
        '''
        通过MCTS输出policy
        '''
        action, actions, pi = self.mcts(state, self.net)   # 网络来提供MCTS所需的东西
        return action, actions, pi
    
    def value(self,
              state):
        '''
        网络直接输出value
        '''
        value = self.net.value_head(state)
        return value