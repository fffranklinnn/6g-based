# train.py

from env import Env
# from PPO import RLAlgorithm
from SAC import RLAlgorithm

def main():
    # 创建环境
    env = Env()

    # 创建RL算法实例
    rl_algorithm = RLAlgorithm(env)

    # 创建模型
    rl_algorithm.create_model()

    # 训练模型
    rl_algorithm.train_model(total_timesteps=10000)

    # 保存模型
    rl_algorithm.save_model()

    # 加载模型
    rl_algorithm.load_model()

    # 评估模型
    rl_algorithm.evaluate_model(num_steps=1000)

if __name__ == "__main__":
    main()
