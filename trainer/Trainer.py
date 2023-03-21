from loss import *


class Trainer():
    '''
    用于训练网络
    '''
    
    def __init__(self,
                 agent,
                 env,
                 **kwargs):
        '''
        初始化一个Trainer.
        net和MCTS已经内含在agent中
        Trinaer对agent.net进行训练
        '''
        self.env = env
        self.agent = agent
        self.examples = []      # 数据库
        #####
        # 传递超参数
        # self.lr = lr
        # ......
        #####
        pass
    
    def sample_examples(self):
        '''
        从数据库中采样
        '''
        pass
    
    def data_augment(self,
                     examples):
        '''
        对采样得到的数据进行增强
        '''
        pass
    
    def learn_one_episode(self,
                          example):
        '''
        对一个元组进行学习
        '''
        # for step in range(max_steps)
        #     mcts.policy()
        #     collect results

        s, a, pi_gt, r_gt = example
        v, _, _, pi = self.agent.net.value_head(s), self.agent.net.policy_head(s)
        
        # Losses.
        # m_loss = mse_loss(...)
        # e_loss = entropy_loss(...)
        # r_loss = regularization_loss(...)
        # loss = self.m_weight * m_loss + self.e_weight * e_loss + self.r_weight * r_loss
        # loss.backward()
        
    def learn(self):
        '''
        训练的主函数
        '''
        # 大致流程如下：
        for iter in range(self.iters):
            # 收集数据
            self.examples.extend(self.play(self.agent))
            # or self.examples.extend(self.env.generate_synthetic_examples())
            examples = self.sample_examples()
            examples = self.data_augment(examples)

            # 此处进行多进程优化
            # todo: 什么时候更新网络参数
            for example in examples:
                self.learn_one_episode(example)

            # 添加logger部分
            
    def play(self, R_limit):
        '''
        进行一次Tensor Game，得到游玩记录
        返回: results
        '''
        results = []
        step = 0
        agent = self.agent
        env = self.env
        env.reset()
        
        while True:
            
            action, actions, pi = agent.policy(env.cur_state)
            cur_state = env.cur_state.copy()
            reward = env.step(action)                          # Will change self.cur_state.
            step += 1
            
            if step >= R_limit:
                reward += env.terminate_reward()
                results.append([cur_state, action, pi, reward])     # Record. (s, a, pi, r).
                break
            
            if env.is_terminate():
                break
                    
            results.append([cur_state, action, pi, reward])     # Record. (s, a, pi, r).
        
        return results