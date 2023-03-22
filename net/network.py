class Torso():
    '''
    网络躯干(Encoder).
    '''
    pass

class PolicyHead():
    pass

class ValueHead():
    pass

class AttentiveModes():
    pass


class Net():
    '''
    网络部分
    '''
    
    def __init__(self,
                 **kwargs):
        '''
        初始化部分
        '''
        # e.g.:
        # self.n_layers = n_layers
        pass
    
    def policy_head(state, self):
        '''
        输出策略
        返回: actions, pi
        '''
        pass
    
    def value_head(state, self):
        '''
        输出效用值
        返回: reward
        '''
        pass