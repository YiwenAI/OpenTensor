import time
import yaml
import copy
import random
import shutil
from tqdm import tqdm 
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join(".")))
from codes.trainer.loss import QuantileLoss
from codes.env import *
from codes.mcts import *
from codes.utils import *
from codes.dataset import *

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
                 lr=5e-3,
                 weight_decay=1e-5,
                 step_size=40000,
                 gamma=.1,
                 a_weight=.5,
                 v_weight=.5,
                 save_freq=10000,
                 self_play_freq=10,
                 self_play_buffer=100000,
                 grad_clip=4.0,
                 val_freq=2000,
                 all_kwargs=None):
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
        
        self.self_examples = []
        self.synthetic_examples = []
        
        self.entropy_loss = torch.nn.CrossEntropyLoss()
        self.quantile_loss = QuantileLoss()
        self.a_weight = a_weight
        self.v_weight = v_weight
        
        self.optimizer = torch.optim.AdamW(net.parameters(),
                                           weight_decay=weight_decay,
                                           lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=step_size,
                                                         gamma=gamma)        
        self.batch_size = batch_size
        self.iters_n = iters_n
        self.grad_clip = grad_clip
        self.save_freq = save_freq
        self.self_play_freq = self_play_freq
        self.self_play_buffer = self_play_buffer
        self.val_freq = val_freq
        
        self.exp_dir = exp_dir
        self.save_dir = os.path.join(exp_dir, exp_name, str(int(time.time())))  
        self.log_dir = os.path.join(self.save_dir, "log")
        
        self.device = device
        self.net.to(device)
        
        self.all_kwargs = all_kwargs
    
    
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
        a_gt = a_gt.long().to(self.device)
        v_gt = v_gt.float().to(self.device)
        
        # Network infer.
        self.net.set_mode("train")
        output = self.net([*s, a_gt])
        o, q = output                   # o: [batch_size, N_steps, N_logits], q: [batch_size, N_quantiles]
        
        # Losses.
        v_loss = self.quantile_loss(q, v_gt)    # v_gt: [batch_size,]
        o = o.transpose(1,2)                    # o: [batch_size, N_logits, N_steps]
        a_loss = self.entropy_loss(o, a_gt)     # a_gt: [batch_size, N_steps], o: [batch_size, N_logits, N_steps]
        loss = self.v_weight * v_loss + self.a_weight * a_loss

        return loss, v_loss, a_loss
    
    
    def val_one_episode(self,
                        episode):
        '''
        对一个元组进行验证, 打印输出
        '''
        
        state, action, reward = episode
        tensor, scalar = state
        
        self.net.set_mode("infer")
        self.net.set_samples_n(4)
        output = self.net(state)
        a, p, _ = output
        a = a.detach().cpu().numpy()
        q, v = self.net.value(output)
        policy, p = self.net.policy(output)
        
        self.net.set_mode("train")
        tensor, scalar, action = tensor[None], scalar[None], action[None]
        state = [tensor, scalar]
        o, _ = self.net([*state, action])
        o = o.detach().cpu().numpy()[0]         # o: [N_steps, N_logits]
        
        log_txt = "\n".join(
            ["\nGt action: \n", str(self.net.logits_to_action(action[0])),
             "\nGt logit: \n", str(action[0]),
            "\nInfer actions: \n", str(policy),
            "\nInfer logits: \n", str(a),
            "\nprob: \n", str(p),
            "\nGt value: \n", str(reward),
            "\nInfer value: \n", str(v),
            "\nquantile: \n", str(q),
            *["\nTop 5 logit for step %d\n: " % step + str(np.argsort(o[step])[-5:]) for step in range(self.net.N_steps)]]
        )
        
        return log_txt
        
        
        
    def learn(self,
              resume=None,
              example_path=None):
        '''
        训练的主函数
        '''
        optimizer = self.optimizer
        scheduler = self.scheduler    
        batch_size = self.batch_size
        self_play_freq = self.self_play_freq
        self_play_buffer = self.self_play_buffer
        
        # Tensorboard.
        os.makedirs(self.save_dir)
        os.makedirs(self.log_dir)
        self.log_writer = SummaryWriter(self.log_dir)
        
        # Save config.
        all_kwargs = self.all_kwargs
        cfg_path = os.path.join(self.save_dir, "config.yaml")
        with open(cfg_path, 'w') as f:
            yaml.dump(all_kwargs, f)
        
        if resume is not None:
            # Load model.
            old_iter = self.load_model(resume)
            # Copy log file.
            old_exp_dir = os.path.join(os.path.dirname(resume), '..')
            # os.system("cp -r %s %s" % (os.path.join(old_exp_dir, 'log', '*'), self.log_dir))
            for log_f in os.listdir(os.path.join(old_exp_dir, "log")):
                shutil.copy(os.path.join(old_exp_dir, "log", log_f), self.log_dir)
        else:
            old_iter = 0
        
        # 1. Get synthetic examples.
        if example_path is not None:
            self.synthetic_examples.extend(self.load_examples(example_path))
        else:
            self.synthetic_examples.extend(self.generate_synthetic_examples(samples_n=3000))
        
        # Dataloader.
        dataset = TupleDataset(S_size=self.S_size,
                               N_steps=self.net.N_steps,
                               coefficients=self.coefficients,
                               self_examples=self.self_examples,
                               synthetic_examples=self.synthetic_examples)
        dataloader = MultiEpochsDataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=32)
        loader = iter(dataloader)
        epoch_ct = 0
        
        for i in tqdm(range(old_iter, self.iters_n)):
            
            # 2. self-play for data.
            # if i % self_play_freq == 0:
            #     self.self_examples.extend(self.play(200 if i < 50000 else 800))

            try:
                batch_example = next(loader)
            except StopIteration:
                # self.self_examples = self.self_examples[-self_play_buffer:]
                # dataset = TupleDataset(S_size=self.S_size,
                #                        N_steps=self.net.N_steps,
                #                        coefficients=self.coefficients,
                #                        self_examples=self.self_examples,
                #                        synthetic_examples=self.synthetic_examples)
                # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
                loader = iter(dataloader)
                batch_example = next(loader)
                print("Epoch: %d finish." % epoch_ct)
                epoch_ct += 1

            # 此处进行多进程优化
            # todo: 什么时候更新网络参数                     
            optimizer.zero_grad()
            loss, v_loss, a_loss = self.learn_one_batch(batch_example)
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.net.parameters(),
                                          max_norm=self.grad_clip)
            optimizer.step()
            scheduler.step()
            
            # 添加logger部分
            if i % 20 == 0:
                # print("Loss: %f, v_loss: %f, a_loss: %f" %
                #       (loss.detach().cpu().item(), v_loss.detach().cpu().item(), a_loss.detach().cpu().item()))
                self.log_writer.add_scalar("loss", loss.detach().cpu().item(), global_step=i)
                self.log_writer.add_scalar("v_loss", v_loss.detach().cpu().item(), global_step=i)
                self.log_writer.add_scalar("a_loss", a_loss.detach().cpu().item(), global_step=i)
            
            if i % self.save_freq == 0:
                ckpt_name = "it%07d.pth" % i
                self.save_model(ckpt_name, i)
                
            if i % self.val_freq == 0:
                val_episode = dataset[random.randint(0, len(dataset)-1)]
                log_txt = self.val_one_episode(val_episode)
                self.log_writer.add_text("Infer", log_txt, global_step=i)
        
        self.save_model("final.pth", i)
            
            
    def play(self,
             simu_times=200) -> list:
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
        mcts.reset(env.cur_state, simulate_times=simu_times)
        
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
              resume=None,
              vis=False):
        
        actions = []
        
        assert resume is not None, "No meaning for random init infer."
        if resume is not None:
            self.load_model(resume)
            exp_dir = os.path.join(os.path.dirname(resume), '..')
            infer_log_dir = os.path.join(exp_dir, "infer")
            os.makedirs(infer_log_dir, exist_ok=True)
            infer_log_f = os.path.join(infer_log_dir, str(int(time.time()))+'.txt')
        
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
            
            action, actions, pi, log_txt = mcts(env.cur_state, net, log=True)
            if vis:
                mcts.visualize()
            print("We choose action(step%d):" % step)
            print(action)
            terminate_flag = env.step(action)                            # Will change self.cur_state. 
            mcts.move(action)                                            # Move MCTS forward.       
            actions.append(action)   
            
            with open(infer_log_f, "a") as f:
                f.write(log_txt)
                f.write("\n\n\n")  
            
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
    
    conf_path = "./config/my_conf.yaml"
    with open(conf_path, 'r', encoding="utf-8") as f:
        kwargs = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    net = Net(**kwargs["net"])
    mcts = MCTS(**kwargs["mcts"],
                init_state=None)
    env = Environment(**kwargs["env"],
                      init_state=None)
    trainer = Trainer(**kwargs["trainer"],
                      net=net, env=env, mcts=mcts,
                      all_kwargs=kwargs)
    
    
    # import pdb; pdb.set_trace()
    # res = trainer.play()
    # res = trainer.generate_synthetic_examples()
    # trainer.learn(example_path="./data/100000_T5_scalar3.npy")
    # import pdb; pdb.set_trace()
    # trainer.load_model("./exp/debug/1680630182/ckpt/it0020000.pth")
    # trainer.infer()
    # trainer.infer(resume="./exp/debug/1680764978/ckpt/it0002000.pth")
    # import pdb; pdb.set_trace()
    # trainer.generate_synthetic_examples(samples_n=3000, save_path="./data/3000_T5_scalar3.npy")
    # trainer.learn(resume=None,
    #               example_path="./data/100000_T5_scalar3.npy")
    # trainer.learn(resume=None)
    trainer.learn(resume=None,
                  example_path="./data/3000_T5_scalar3.npy")