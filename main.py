import yaml

from codes.env import Environment
from codes.mcts import MCTS
from codes.net import Net
from codes.trainer import Trainer


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
    
    
    # import pdb; pdb.set_trace()
    # res = trainer.play()
    # res = trainer.generate_synthetic_examples()
    # trainer.learn()
    # import pdb; pdb.set_trace()
    # trainer.load_model("./exp/debug/1680630182/ckpt/it0020000.pth")
    # trainer.infer()
    # trainer.infer(resume="./exp/debug/1680764978/ckpt/it0002000.pth")
    # import pdb; pdb.set_trace()
    # trainer.generate_synthetic_examples(samples_n=1000000, save_path="./data/1000000_S4T7_scalar3.npy")
    # trainer.play(play_times=10000, save_path="./data/self-10000_T5_scalar3.npy")
    # trainer.learn(resume="./exp/first_exp/1681827145/ckpt/it0750000.pth",
    #               example_path="./data/100000_T5_scalar3.npy",
    #               only_weight=True)
    trainer.learn(resume="./exp/S4T7_exp2/1682823074/ckpt/it0600000.pth",
                  example_path="./data/100000_S4T7_scalar3.npy")
    # trainer.infer(resume="./exp/first_exp/1682068899/ckpt/it1500000.pth",
    #               mcts_samples_n=32,
    #               mcts_simu_times=65536,
    #               vis=False)
