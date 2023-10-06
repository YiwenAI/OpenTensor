from codes.env import *
from typing import List

class ENVS():
    
    def __init__(self,
                 env_list: List[Environment]):
        
        self.env_list = env_list
        self.env_n = len(env_list)
        self.terminate_list = [False] * self.env_n        
        
    
    def reset(self):
        
        for env in self.env_list:
            env.reset()
        self.terminate_list = [False] * self.env_n
        
    
    def step(self,
             action_list):
        
        for idx, (env, action) in enumerate(zip(self.env_list, action_list)):
            if not self.terminate_list[idx]:
                self.terminate_list[idx] = env.step(action)
             
    
    def get_curstates(self):
        
        state_list = []
        for env in self.env_list:
            state_list.append(env.cur_state.copy())
        
        return state_list
    
    
    def is_all_terminated(self):
        
        flag = True
        for sub_flag in self.terminate_list:
            flag = flag & sub_flag
            
        return flag
    
    
    def get_rewards(self):
        
        reward_list = []
        for env in self.env_list:
            reward_list.append(env.accumulate_reward)
            
        return reward_list
    
    
    def get_stepcts(self):
        
        step_ct_list = []
        for env in self.env_list:
            step_ct_list.append(env.step_ct)
            
        return step_ct_list