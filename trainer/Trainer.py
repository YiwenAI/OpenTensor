import random
from tqdm import tqdm 
import torch
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join(".")))
from loss import QuantileLoss
from net import *
from env import *
from mcts import *
from utils import *

class Trainer():
    '''
    用于训练网络
    '''
    
    def __init__(self,
                 net: Net,
                 env: Environment,
                 mcts: MCTS,
                 S_size=4,
                 T=7,
                 coefficients=[0, 1, -1],
                 batch_size=64,
                 iters_n=50000,
                 **kwargs):
        '''
        初始化一个Trainer.
        包含net, env和MCTS
        '''
        self.env = env
        self.net = net
        self.mcts = mcts
        self.S_size = S_size
        self.T = T
        self.coefficients = coefficients
        
        self.examples = []      # 数据库
        
        self.entropy_loss = torch.nn.CrossEntropyLoss()
        self.quantile_loss = QuantileLoss()
        self.a_weight = 1
        self.v_weight = 1
        
        self.optimizer = torch.optim.AdamW(net.parameters(),
                                           weight_decay=1e-5,
                                           lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=500000,
                                                         gamma=.1)
        self.batch_size = batch_size
        self.iters_n = iters_n


    
    def sample_examples(self,
                        samples_n) -> list:
        '''
        从数据库中采样
        '''
        return random.sample(self.examples, samples_n)
    
    def data_augment(self,
                     examples):
        '''
        对采样得到的数据进行增强
        '''
        pass
    
    def generate_synthetic_examples(self,
                                    prob=[.8, .1, .1],
                                    samples_n=5000,
                                    R_limit=12) -> list:
        '''
        生成人工合成的Tensor examples
        返回: results
        '''
        S_size = self.S_size
        coefficients = self.coefficients
        T = self.T
        
        total_results = []
        for _ in (range(samples_n)):
            R = random.randint(1, R_limit)
            sample = np.zeros((S_size, S_size, S_size), dtype=np.int32)
            states = []        
            actions = []
            rewards = []
            for r in range(1, (R+1)):
                ct = 0
                while True:
                    u = np.random.choice(coefficients, size=(S_size,), p=prob, replace=True)
                    v = np.random.choice(coefficients, size=(S_size,), p=prob, replace=True)
                    w = np.random.choice(coefficients, size=(S_size,), p=prob, replace=True)
                    ct += 1
                    if not is_zero_tensor(outer(u, v, w)):
                        break
                    if ct > 100000:
                        raise Exception("Oh my god...")
                sample = sample + outer(u, v, w)
                action = np.stack([u, v, w], axis=0)
                actions.append(action)
                states.append(sample.copy())
                rewards.append(-r)
                
            # Reformulate the results.
            states.reverse(); actions.reverse(); rewards.reverse()
            actions_tensor = [action2tensor(action) for action in actions]
            for idx, state in enumerate(states):
                tensors = np.zeros((T, S_size, S_size, S_size), dtype=np.int32)
                tensors[0] = state            # state.
                if idx != 0:
                    # History actions.
                    tensors[1:(idx+1)] = np.stack(reversed(actions_tensor[max(idx-(T-1), 0):idx]), axis=0)        
                scalars = np.array([idx, idx, idx])     #FIXME: Havn't decided the scalars.
                
                cur_state = [tensors, scalars]
                action = actions[idx]
                reward = rewards[idx]
                total_results.append([cur_state, action, reward])
                
        return total_results
        
    
    def learn_one_episode(self,
                          example) -> torch.autograd.Variable:
        '''
        对一个元组进行学习
        '''
        
        # Groundtruth.
        s, a_gt, v_gt = example
        a_gt = torch.tensor(self.net.action_to_logits(a_gt)).long()
        
        # Network infer.
        self.net.set_mode("train")
        output = self.net([*s, a_gt])
        o, q = output                   # o: [N_steps, N_logits], q: [N_quantiles]
        
        # Losses.
        v_loss = self.quantile_loss(q, v_gt)    # v_gt: scalar.
        a_loss = self.entropy_loss(o, a_gt)     # a_gt: [N_logits,]
        loss = self.v_weight * v_loss + self.a_weight * a_loss

        return loss
        
        
    def learn(self):
        '''
        训练的主函数
        '''
        optimizer = self.optimizer
        scheduler = self.scheduler
        batch_size = self.batch_size
        
        # 1. Get synthetic examples.
        self.examples.extend(self.generate_synthetic_examples(samples_n=3000))
        
        for iter in tqdm(range(self.iters_n)):
            # 2. self-play for data.
            # self.examples.extend(self.play(self.net))

            examples = self.sample_examples(samples_n=batch_size)

            # 此处进行多进程优化
            # todo: 什么时候更新网络参数
            optimizer.zero_grad()
            for example in examples:
                loss = self.learn_one_episode(example) / batch_size
                loss.backward(retain_graph=True)          #FIXME: Batch size right?
            optimizer.step()
            scheduler.step()
            
            # 添加logger部分
            if iter % 10 == 0:
                print("Loss: %f" % loss.detach().cpu().item())
            
            
    def play(self) -> list:
        '''
        进行一次Tensor Game, 得到游玩记录
        返回: results
        '''
        results = []
        net = self.net
        env = self.env
        mcts = self.mcts
        env.reset()
        net.set_mode("infer")
        mcts.reset(env.cur_state)
        
        while True:
            
            action, actions, pi = mcts(env.cur_state, net)
            cur_state = env.get_network_input()                        # [tensors, scalars]
            terminate_flag = env.step(action)                          # Will change self.cur_state. 
            mcts.move(action)                                          # Move MCTS forward.    
            results.append([cur_state, action])                        # Record. (s, a).
            # print("Step forward %d" % env.step_ct)
            
            if terminate_flag:
                final_reward = env.accumulate_reward
                for step in range(env.step_ct):
                    results[step] += [final_reward + step]             # Final results. (s, a, r(s)).
                    # Note:
                    # a is not included in the history actions of s.
                return results
            
            
            
if __name__ == '__main__':
    
    net = Net(N_samples=2,
              T=3, N_steps=3,
              n_attentive=4, N_heads=16, N_features=16)
    mcts = MCTS(simulate_times=20,
                init_state=None)
    env = Environment(S_size=4,
                      R_limit=8, T=3)
    
    trainer = Trainer(net=net, env=env, mcts=mcts, T=3)
    
    # res = trainer.play()
    # res = trainer.generate_synthetic_examples()
    trainer.learn()
    import pdb; pdb.set_trace()