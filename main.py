import yaml

from codes.env import Environment
from codes.mcts import MCTS
from codes.net import Net
from codes.trainer import Trainer, Player


def handle_config():
    pass


# todo: alphastrassen 时间消耗分析
# todo: 建议写单元测试
# todo: 有无高效mcts代码 如何进行分布式设计   AlphaZero MPI
if __name__ == '__main__':
    
    # conf_path = "./config/my_conf.yaml"
    conf_path = "./config/S_4.yaml"
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
    

    # trainer.learn(resume=None,
    #               example_path="./data/traj_data/100000_S4T7_scalar3_filtered.npy",
    #               self_example_path=None)
    
    # trainer.generate_synthetic_examples(samples_n=100000,
    #                                     save_path="./data/traj_data/100000_S4T7_scalar3_filtered.npy")
    # trainer.learn(resume="./exp/S4T7_selfplay/1685326378/ckpt/latest.pth",
    #               example_path="./data/traj_data/100000_S4T7_scalar3_filtered.npy",
    #               self_example_path="./exp/S4T7_selfplay/1685326378/data/total_self_data.npy",
    #               self_play=True)
    # while True:
    #     trainer.infer(resume="./exp/S4T7_exp4/1685438936/ckpt/latest.pth",
    #                 mcts_samples_n=32,
    #                 mcts_simu_times=65536,
    #                 vis=False,
    #                 noise=False)
        
    # import numpy as np
    # while True:
    #     trainer.infer(resume="./exp/S4T7_exp3/1683894384/ckpt/it2200000.pth",
    #                 mcts_samples_n=32,
    #                 mcts_simu_times=65536 * 2,
    #                 vis=False,
    #                 init_state=np.array([
    #                     [[0,0,0,-1], [0,1,0,0], [0,0,0,0], [-1,0,0,-1]],
    #                     [[0,0,0,0], [0,0,0,0], [1,0,0,0], [0,1,0,0]],
    #                     [[0,0,1,0], [0,0,0,1], [0,0,0,0], [0,0,0,0]],
    #                     [[-1,0,0,-1], [0,0,0,0], [0,0,1,0], [-1,0,0,0]]
    #                 ]),
    #                 no_base_change=False)  
    
    # trainer.filter_train_data(n=100,
    #                           resume="./exp/S4T7_exp3/1684058680/ckpt/it2250000.pth",
    #                           example_path="./data/traj_data/10000_S4T7_scalar3.npy",
    #                           mcts_samples_n=32,
    #                           mcts_simu_times=4096)
    
    self_play_net = Net(**kwargs["net"])
    player = Player(net=self_play_net,
                    env=env,
                    mcts=mcts,
                    exp_dir="./exp/S4T7_selfplay/1685590684",
                    simu_times=800,
                    play_times=1,
                    num_workers=64,
                    device="cuda:0",
                    noise=True)
    player.run()     # Running forever...    