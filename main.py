from env import Environment
from agent import AlphaTensor
from mcts import MCTS
from net import Net
from trainer import Trainer


def handle_config():
    pass


# todo: alphastrassen 时间消耗分析
# todo: 建议写单元测试
# todo: 有无高效mcts代码 如何进行分布式设计   AlphaZero MPI
if __name__ == '__main__':

    # argparse handle config
    # 这样可以保留多套超参数
    S_size = 4
    R_limit = 8
    mode = "train"
    # and so on...

    handle_config()

    # Initialize
    env = Environment(S_size, R_limit)  # 至少需要使用vec_env
    # 注意，MCTS和环境绑定，vec_env应该在mcts内进行
    # todo: check stable_baselines3 vec_env 机制
    # todo: rllib 如何进行多进程训练 worker
    
    net = Net()  # 正常torch写的网络，可以使用多套方案

    # 建议将alphatensor和trainer合并
    agent = AlphaTensor(net)
    
    trainer = Trainer(agent,
                      env)

    if mode == "train":
        trainer.learn()
        
    elif mode == "test":
        trainer.play()