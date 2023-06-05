import copy
from torch.utils.tensorboard import SummaryWriter
import sys
import os
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join(".")))
from codes.trainer.loss import QuantileLoss
from codes.env import *
from codes.mcts import *
from codes.utils import *
from codes.dataset import *
from codes.multi_runner import *


class Player():

    def __init__(self,
                 net, env, mcts,
                 exp_dir,
                 simu_times=400,
                 play_times=10,
                 num_workers=256,
                 device="cuda:1",
                 noise=False):
        
        self.net = net
        self.env = env
        self.mcts = mcts
        
        net.to(device)
        
        self.exp_dir = exp_dir
        self.trainer_logger = SummaryWriter(log_dir=os.path.join(exp_dir, "log"))
        
        self.simu_times = simu_times
        self.play_times = play_times
        self.num_workers = num_workers
        self.device = device
        self.noise = noise
        
        self.call_ct = 0
        

    def play(self, warm_up=False) -> list:
        '''
        进行一次Tensor Game, 得到游玩记录
        返回: results
        '''
        
        num_workers = self.num_workers
        simu_times = self.simu_times
        play_times = self.play_times
        noise = self.noise
        
        if warm_up:
            simu_times = 40
            play_times = 1
            num_workers = 10
        
        results = []
        avg_steps = 0
        net = self.net
        env = self.env
        mcts = self.mcts
        net.set_mode("infer")
        net.eval()
        
        # wkdir.
        save_dir = os.path.join(self.exp_dir, "data")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "self_data.npy")
        
        # Load ckpt.
        latest_path = os.path.join(self.exp_dir, "ckpt", "latest.pth")
        ct = 0
        while True:
            if ct > 10000000:
                raise Exception
            try:
                self.load_model(ckpt_path=latest_path, to_device=self.device)
                break
            except:
                ct += 1
                continue
        
        # Multi runner.
        env_list = [copy.deepcopy(env) for _ in range(num_workers)]
        mcts_list = [copy.deepcopy(mcts) for _ in range(num_workers)]
        envs = ENVS(env_list)
        mctsf = MCTSF(mcts_list, simulate_times=simu_times)
        
        for game in (range(play_times)):
            
            envs.reset()
            state_list = envs.get_curstates()
            mctsf.reset(state_list)
            trajs = []            
            
            while True:
                state_list = envs.get_curstates() 
                action_list, _, _ = mctsf(state_list, net, noise=noise)                          
                envs.step(action_list)
                terminate_flag = envs.is_all_terminated()
                mctsf.move(action_list)                                          # Move MCTS forward.    
                # one_result.append([state_list, action_list])                        # Record. (s, a).
                trajs.append([[state, action] for state, action in zip(state_list, action_list)])
                
                if terminate_flag:
                    reward_list = envs.get_rewards()
                    # for step in range(env.step_ct):
                    #     one_result[step] += [final_reward + step]             # Final results. (s, a, r(s)).
                    #     # Note:
                    #     # a is not included in the history actions of s.
                    for step, _trajs in enumerate(trajs):
                        for idx, traj_state in enumerate(_trajs):     # traj_state: [state, action]
                            traj_state += [reward_list[idx] + (step)]
                    
                    # Prepare per traj.
                    for idx in range(num_workers):
                        one_traj = [_trajs[idx] for _trajs in trajs]     # [ [s1, a1, r1], [s2, a2, r2] ... ]
                        one_traj = [episode for episode in one_traj if not is_zero_tensor(episode[0])]   # Filter the invalid state.
                        
                        states, actions, rewards = \
                            [episode[0] for episode in one_traj], \
                            [episode[1] for episode in one_traj], \
                            [episode[2] for episode in one_traj]
                        
                        one_traj = [states, actions, rewards]         
                        results.append(one_traj)      
                    
                    step_ct_list = envs.get_stepcts()
                    batch_avg_step_ct = np.array(step_ct_list).mean()

                    avg_steps = (game / (game+1)) * avg_steps + batch_avg_step_ct / (game+1)
                    break
        
        # Insert a bubble.
        # results: (n_trajs, 3, traj_length*)
        one_state, one_action, one_reward = results[-1][0][0], results[-1][1][0], results[-1][2][0]
        bubble_traj = [[one_state,], [one_action,], [one_reward,]]
        results.append(bubble_traj)
        if save_path is not None:
            results = np.array(results, dtype=object)   # (n_trajs, 3, traj_length*). *traj_length is not fixed. Decompose order...
            np.save(save_path, results)
        print("Avg step is: %f" % avg_steps)
        
        self.trainer_logger.add_text("Self-play", str(one_traj), global_step=self.call_ct)
        self.call_ct += 1
            
        return results, one_traj   
    
    
    def run(self):
        
        self.play(warm_up=True)
        while True:
            self.play()
            print("Finish playing!")
            
            
    def load_model(self, ckpt_path, only_weight=False, to_device="cuda:0"):
        ckpt = torch.load(ckpt_path)
        self.net.load_state_dict(ckpt['model'])
        return ckpt['iter']