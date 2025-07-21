import yaml
import argparse

from codes.env import Environment
from codes.mcts import MCTS
from codes.net import Net
from codes.trainer import Trainer, Player

def parse():
    parser = argparse.ArgumentParser(description="OpenTensor")
    parser.add_argument('--config', type=str, default="./config/S_4.yaml")
    parser.add_argument('--mode', type=str, default="train", help="modes: [generate_data, train, infer]")
    parser.add_argument('--resume', default=None, help="resume ckpt path for training")
    parser.add_argument('--run_dir', default=None, help="ckpt path for inference")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    conf_path = args.config
    mode = args.mode
    resume = args.resume

    with open(conf_path, 'r', encoding="utf-8") as f:
        kwargs = yaml.load(f.read(), Loader=yaml.FullLoader)

    # === Automatically pass projection parameters ===
    # If projection is written in config, add it to net's kwargs
    if "projection_dim" in kwargs["net"]:
        kwargs["net"]["use_projection"] = True  # Forced to open
    else:
        kwargs["net"]["use_projection"] = False

    # === Instantiating Net will register the projection ===
    net = Net(**kwargs["net"])
    mcts = MCTS(**kwargs["mcts"], init_state=None)
    env = Environment(**kwargs["env"], init_state=None)
    trainer = Trainer(**kwargs["trainer"],
                      net=net, env=env, mcts=mcts,
                      all_kwargs=kwargs)

    S_size = kwargs["env"]["S_size"]
    T = kwargs["env"]["T"]

    if mode == "generate_data":
        trainer.generate_synthetic_examples(
            samples_n=100000,
            save_path="./data/100000_S%dT%d_scalar3_filtered.npy" % (S_size, T)
        )

    elif mode == "train":
        trainer.learn(
            resume=resume,
            example_path="./data/100000_S%dT%d_scalar3_filtered.npy" % (S_size, T),
            self_example_path=None
        )

    elif mode == "infer":
        assert args.run_dir is not None, "Please specify --run_dir to the checkpoint you want to test!"
        trainer.infer(resume=args.run_dir)
