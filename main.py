import yaml
import argparse
import argparse

from codes.env import Environment
from codes.mcts import MCTS
from codes.net import Net
from codes.trainer import Trainer, Player


def parse():
    parser = argparse.ArgumentParser(description="OpenTensor")
    parser.add_argument('--config', type=str, default="./config/S_4.yaml")
    parser.add_argument('--mode', type=str, default="train", help="three modes: [generate_data, train, infer]")
    parser.add_argument('--resume', default=None, help="resume ckpt path")
    parser.add_argument('--run_dir', default="./exp/S4T7_selfplay/1685590684", help="The run dir to infer")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = parse()
    conf_path = args.config
    mode = args.mode
    resume = args.resume
    
    args = parse()
    conf_path = args.config
    mode = args.mode
    resume = args.resume
    
    with open(conf_path, 'r', encoding="utf-8") as f:
        kwargs = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    net = Net(**kwargs["net"])
    mcts = MCTS(kwargs=kwargs["mcts"],
                init_state=None)
    env = Environment(kwargs=kwargs["env"],
                      init_state=None)
    trainer = Trainer(kwargs=kwargs["trainer"],
                      net=net, env=env, mcts=mcts,
                      all_kwargs=kwargs)
    
    S_size = kwargs["env"]["S_size"]
    T = kwargs["env"]["T"]    
    if mode == "generate_data":
        trainer.generate_synthetic_examples(samples_n=100,
                                            save_path="./data/100000_S%dT%d_scalar3_filtered.npy" % (S_size, T))    
    elif mode == "train":
        trainer.learn(resume=resume,
                    example_path="./data/100000_S%dT%d_scalar3_filtered.npy" % (S_size, T),
                    self_play=False)
    
    elif mode == "infer":
        self_play_net = Net(**kwargs["net"])
        player = Player(net=self_play_net,
                        env=env,
                        mcts=mcts,
                        exp_dir=args.run_dir,
                        simu_times=800,
                        play_times=1,
                        num_workers=64,
                        device="cuda:0",
                        noise=False)
        player.run()     # Running forever...       