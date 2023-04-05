import time
import random
from tqdm import tqdm 
import torch
from torch.utils.tensorboard import SummaryWriter
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
                 batch_size=1024,
                 iters_n=50000,
                 exp_dir="exp",
                 exp_name="debug",
                 device="cuda",
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
        self.a_weight = .5
        self.v_weight = .5
        
        self.optimizer = torch.optim.AdamW(net.parameters(),
                                           weight_decay=1e-5,
                                           lr=5e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=40000,
                                                         gamma=.1)        
        self.batch_size = batch_size
        self.iters_n = iters_n
        
        self.exp_dir = exp_dir
        self.save_dir = os.path.join(exp_dir, exp_name, str(int(time.time())))  
        self.log_dir = os.path.join(self.save_dir, "log")
        
        self.device = device
        self.net.to(device)

    
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
                                    samples_n=10000,
                                    R_limit=12,
                                    save_path=None) -> list:
        '''
        生成人工合成的Tensor examples
        返回: results
        '''
        S_size = self.S_size
        coefficients = self.coefficients
        T = self.T
        
        total_results = []
        for _ in tqdm(range(samples_n)):
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
                
                
        if save_path is not None:
            np.save(save_path, total_results)
            
        return total_results
        
    
    def learn_one_batch(self,
                        batch_example) -> torch.autograd.Variable:
        '''
        对一个元组进行学习
        '''
        
        # Groundtruth.
        s, a_gt, v_gt = batch_example     # s: [tensor, scalar]
        a_gt = [canonicalize_action(a) for a in a_gt]
        a_gt = torch.tensor(np.stack([self.net.action_to_logits(a) for a in a_gt], axis=0)).long().to(self.device)
        v_gt = torch.tensor(v_gt).float().to(self.device)
        
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
        
        
    def learn(self,
              resume=None,
              example_path=None):
        '''
        训练的主函数
        '''
        optimizer = self.optimizer
        scheduler = self.scheduler    
        batch_size = self.batch_size
        
        os.makedirs(self.save_dir)
        os.makedirs(self.log_dir)
        self.log_writer = SummaryWriter(self.log_dir)
        
        if resume is not None:
            old_iter = self.load_model(resume)
        else:
            old_iter = 0
        
        # 1. Get synthetic examples.
        if example_path is not None:
            self.examples.extend(self.load_examples(example_path))
        else:
            self.examples.extend(self.generate_synthetic_examples(samples_n=100000))
        
        for iter in tqdm(range(old_iter, self.iters_n)):
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
            if iter % 20 == 0:
                # print("Loss: %f, v_loss: %f, a_loss: %f" %
                #       (loss.detach().cpu().item(), v_loss.detach().cpu().item(), a_loss.detach().cpu().item()))
                self.log_writer.add_scalar("loss", loss.detach().cpu().item(), global_step=iter)
                self.log_writer.add_scalar("v_loss", v_loss.detach().cpu().item(), global_step=iter)
                self.log_writer.add_scalar("a_loss", a_loss.detach().cpu().item(), global_step=iter)
            
            if iter % 10000 == 0:
                ckpt_name = "it%07d.pth" % iter
                self.save_model(ckpt_name, iter)
        
        self.save_model("final.pth", iter)
            
            
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
        net.eval()
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
              mcts_simu_times=10000,
              mcts_samples_n=16,
              step_limit=12,
              resume=None):
        
        actions = []
        
        if resume is not None:
            self.load_model(resume)
        
        net = self.net
        env = self.env
        mcts = self.mcts
        
        env.reset(no_base_change=True)
        net.set_mode("infer")
        net.set_samples_n(mcts_samples_n)
        net.eval()
        mcts.reset(env.cur_state, simulate_times=mcts_simu_times)
        env.R_limit = step_limit + 1
        
        for step in tqdm(range(step_limit)):
            print("Current state is (step%d):" % step)
            print(env.cur_state)
            
            action, actions, pi = mcts(env.cur_state, net)
            print("We choose action(step%d):" % step)
            print(action)
            terminate_flag = env.step(action)                            # Will change self.cur_state. 
            mcts.move(action)                                            # Move MCTS forward.       
            actions.append(action)     
            
            if terminate_flag:
                print("We get to the end!")
                
            
        print("Final result:")
        print(env.cur_state)
        
        print("Actions are:")
        print(np.stack(actions, axis=0))
        
    
    def save_model(self, ckpt_name, iter):
        save_dir = os.path.join(self.save_dir, "ckpt")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, ckpt_name)
        torch.save({'model': self.net.state_dict(),
                    'iter': iter,
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()}, save_path)


    def load_model(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        self.net.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        return ckpt['iter']
    
    
    def load_examples(self, example_path):
        return np.load(example_path, allow_pickle=True)
        
            
if __name__ == '__main__':
    
    net = Net(N_samples=2,
              T=5, N_steps=6,
              n_attentive=4, N_heads=16, N_features=16,
              device='cuda')
    mcts = MCTS(simulate_times=20,
                init_state=None)
    env = Environment(S_size=4,
                      R_limit=8, T=5)
    
    trainer = Trainer(net=net, env=env, mcts=mcts, T=5,
                      save_dir="ckpt/debug",
                      device='cuda')
    
    
    # import pdb; pdb.set_trace()
    # res = trainer.play()
    # res = trainer.generate_synthetic_examples()
    # trainer.learn()
    # import pdb; pdb.set_trace()
    # trainer.load_model("./exp/debug/1680630182/ckpt/it0020000.pth")
    # trainer.infer()
    trainer.infer(resume="./exp/debug/1680673032/ckpt/it0030000.pth")
    # import pdb; pdb.set_trace()
    # trainer.generate_synthetic_examples(samples_n=100000, save_path="./data/100000_T5_scalar3.npy")
    # trainer.learn(resume="./exp/debug/1680630182/ckpt/it0020000.pth",
                  # example_path="./data/100000_T5_scalar3.npy")