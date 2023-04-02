import random
from tqdm import tqdm 
import torch
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join(".")))
from loss import QuantileLoss
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
                 save_dir="ckpt",
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
        self.v_weight = .1
        
        self.optimizer = torch.optim.AdamW(net.parameters(),
                                           weight_decay=1e-5,
                                           lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=500000,
                                                         gamma=.1)
        self.batch_size = batch_size
        self.iters_n = iters_n
        
        self.save_dir = save_dir


    
    def sample_examples(self,
                        samples_n) -> list:
        '''
        从数据库中采样
        '''
        sampled = random.sample(self.examples, samples_n)
        # Merge example into a batch.
        batch_tensor = np.stack([episode[0][0] for episode in sampled], axis=0)
        batch_scalar = np.stack([episode[0][1] for episode in sampled], axis=0)
        batch_action = np.stack([episode[1] for episode in sampled], axis=0)
        batch_value = np.stack([episode[2] for episode in sampled], axis=0)
        
        return [batch_tensor, batch_scalar], batch_action, batch_value
        
    
    def data_augment(self,
                     examples):
        '''
        对采样得到的数据进行增强
        '''
        pass
    
    def generate_synthetic_examples(self,
                                    prob=[.8, .1, .1],
                                    samples_n=100000,
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
        
    
    def learn_one_batch(self,
                        batch_example) -> torch.autograd.Variable:
        '''
        对一个元组进行学习
        '''
        
        # Groundtruth.
        s, a_gt, v_gt = batch_example     # s: [tensor, scalar]
        a_gt = torch.tensor(np.stack([self.net.action_to_logits(a) for a in a_gt], axis=0)).long()
        v_gt = torch.tensor(v_gt).float()
        
        # Network infer.
        self.net.set_mode("train")
        output = self.net([*s, a_gt])
        o, q = output                   # o: [batch_size, N_steps, N_logits], q: [batch_size, N_quantiles]
        
        # Losses.
        v_loss = self.quantile_loss(q, v_gt)    # v_gt: [batch_size,]
        o, a_gt = o.reshape((-1, o.shape[-1])), a_gt.reshape((-1,))    # o: [batch_size*N_steps, N_logits]
        a_loss = self.entropy_loss(o, a_gt)     # a_gt: [batch_size*N_steps]
        loss = self.v_weight * v_loss + self.a_weight * a_loss

        return loss, v_loss, a_loss
        
        
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

            batch_example = self.sample_examples(samples_n=batch_size)

            # 此处进行多进程优化
            # todo: 什么时候更新网络参数
            optimizer.zero_grad()
            loss, v_loss, a_loss = self.learn_one_batch(batch_example)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # 添加logger部分
            if iter % 100 == 0:
                print("Loss: %f, v_loss: %f, a_loss: %f" %
                      (loss.detach().cpu().item(), v_loss.detach().cpu().item(), a_loss.detach().cpu().item()))
            
            if iter % 10000 == 0:
                ckpt_name = "it%07d.pth" % iter
                self.save_model(ckpt_name)
        
        self.save_model("final.pth")
            
            
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
            state_input = env.get_network_input()                        # [tensors, scalars]
            terminate_flag = env.step(action)                            # Will change self.cur_state. 
            mcts.move(action)                                            # Move MCTS forward.    
            results.append([state_input, action])                        # Record. (s, a).
            # print("Step forward %d" % env.step_ct)
            
            if terminate_flag:
                final_reward = env.accumulate_reward
                for step in range(env.step_ct):
                    results[step] += [final_reward + step]             # Final results. (s, a, r(s)).
                    # Note:
                    # a is not included in the history actions of s.
                return results
            
    
    def infer(self,
              mcts_simu_times=400,
              mcts_samples_n=16,
              step_limit=12):
        
        actions = []
        
        net = self.net
        env = self.env
        mcts = self.mcts
        
        env.reset(no_base_change=True)
        net.set_mode("infer")
        net.set_samples_n(mcts_samples_n)
        mcts.reset(env.cur_state, simulate_times=mcts_simu_times)
        env.R_limit = step_limit
        
        for step in tqdm(range(step_limit)):
            print("Current state is (step%d):" % step)
            print(env.cur_state)
            
            action, actions, pi = mcts(env.cur_state, net)
            print("We choose action(step%d):" % step)
            print(action)
            terminate_flag = env.step(action)                            # Will change self.cur_state. 
            mcts.move(action)                                            # Move MCTS forward.       
            actions.append(action)     
            
        print("Final result:")
        print(env.cur_state)
        
        print("Actions are:")
        print(np.stack(actions, axis=0))
        
    
    def save_model(self, ckpt_name):
        save_path = os.path.join(self.save_dir, ckpt_name)
        torch.save({'model': self.net.state_dict()}, save_path)


    def load_model(self, ckpt_path):
        state_dict = torch.load(ckpt_path)
        self.net.load_state_dict(state_dict['model'])
            
            
if __name__ == '__main__':
    
    net = Net(N_samples=2,
              T=3, N_steps=3,
              n_attentive=4, N_heads=16, N_features=16)
    mcts = MCTS(simulate_times=20,
                init_state=None)
    env = Environment(S_size=4,
                      R_limit=8, T=3)
    
    trainer = Trainer(net=net, env=env, mcts=mcts, T=3)
    # import pdb; pdb.set_trace()
    # res = trainer.play()
    # res = trainer.generate_synthetic_examples()
    # trainer.learn()
    # import pdb; pdb.set_trace()
    trainer.load_model("./ckpt/it0050000.pth")
    trainer.infer()
    