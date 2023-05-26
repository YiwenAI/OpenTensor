# MCTSF: Monto Colo Tree Search Forest.
from codes.mcts import *
from typing import List

class MCTSF():
    '''
    多个蒙特卡洛树的模拟
    '''
    
    def __init__(self,
                 mcts_list: List[MCTS],
                 simulate_times=400):
        
        self.mcts_list = mcts_list
        self.mcts_n = len(mcts_list)
        self.simulate_times = simulate_times
        
    
    def __call__(self,
                 state_list,
                 net: Net):
        
        # Check the states.
        for mcts, state in zip(self.mcts_list, state_list):
            assert is_equal(state, mcts.root_node.state), "State is inconsistent."
            
        # Simu.
        for simu in tqdm(range(self.simulate_times)):
            # Select the leaf nodes.
            node_list = []
            for mcts in self.mcts_list:
                node = mcts.root_node
                while not node.is_leaf:
                    node, scores = node.select()
                node_list.append(node)
                
            # Get network input to expand...
            node: Node
            batch_tensors, batch_scalars = [], []
            for node in node_list:
                tensors, scalars = node.get_network_input(net)
                batch_tensors.append(tensors); batch_scalars.append(scalars)
            batch_tensors, batch_scalars = np.array(batch_tensors), np.array(batch_scalars)
            # Infer...
            net.set_mode("infer")
            with torch.no_grad():
                batch_output = net([batch_tensors, batch_scalars])
                _, batch_value, batch_policy, prob = *net.value(batch_output), *net.policy(batch_output)
                # batch_value: [B,]   batch_policy: [B, N_samples, 3, S_size]
                del batch_output, batch_tensors, batch_scalars
            batch_policy = [[canonicalize_action(action) for action in policy] for policy in batch_policy]
            
            # Expand...
            for node, value, policy in zip(node_list, batch_value, batch_policy):
                node.expand(net, network_output=(value, policy))
        
        # Get results.
        actions_list = [mcts.root_node.actions for mcts in self.mcts_list]    
        N_list = [mcts.root_node.N for mcts in self.mcts_list]
        visit_ratio_list = [(np.array(N) / sum(N)).tolist() for N in N_list]
        action_list = [actions_list[idx][np.argmax(visit_ratio_list[idx])] for idx in range(self.mcts_n)]
        
        return action_list, actions_list, visit_ratio_list
        
        
    def reset(self,
              state_list):
        
        for mcts, state in zip(self.mcts_list, state_list):
            mcts.reset(state, simulate_times=self.simulate_times)
            
    
    def move(self,
             action_list):
        
        for mcts, action in zip(self.mcts_list, action_list):
            mcts.move(action)        