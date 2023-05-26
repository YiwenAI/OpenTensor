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
    

    # trainer.learn(resume="./exp/first_exp/1681827145/ckpt/it0750000.pth",
    #               example_path="./data/100000_T5_scalar3.npy",
    #               only_weight=True)
    # trainer.learn(resume=None,
    #               example_path="./data/traj_data/100000_S4T7_scalar3.npy",
    #               self_example_path=None,
    #               self_play=True)
    # while True:
    #     trainer.infer(resume="./exp/S4T7_selfplay/1684844835/ckpt/it0775000.pth",
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
    
    self_play_net = Net(**kwargs["net"])
    player = Player(net=self_play_net,
                    env=env,
                    mcts=mcts,
                    exp_dir="./exp/S4T7_selfplay/1685088597",
                    simu_times=400,
                    play_times=1,
                    num_workers=64,
                    device="cuda:0")
    player.run()     # Running forever...    