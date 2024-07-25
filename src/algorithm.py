# algorithm.py

from stable_baselines3 import PPO
# from env import env

class RLAlgorithm:
    def __init__(self, env, model_path="ppo_env"):
        """
        初始化RL算法类。

        参数:
        env -- 强化学习环境
        model_path -- 模型保存路径
        """
        self.env = env
        self.model_path = model_path
        self.model = None

    def create_model(self):
        """
        创建并返回一个PPO模型。
        """
        self.model = PPO("MlpPolicy", self.env, verbose=1)
        return self.model

    def train_model(self, total_timesteps=10000):
        """
        训练模型。

        参数:
        total_timesteps -- 训练的总步数
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        self.model.learn(total_timesteps=total_timesteps)
        return self.model

    def save_model(self):
        """
        保存模型到指定路径。
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        self.model.save(self.model_path)

    def load_model(self):
        """
        从指定路径加载模型。
        """
        self.model = PPO.load(self.model_path)
        return self.model

    def evaluate_model(self, num_steps=1000):
        """
        评估模型在环境中的表现。

        参数:
        num_steps -- 评估步数
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() or load_model() first.")
        obs = self.env.reset()
        for i in range(num_steps):
            action, _states = self.model.predict(obs)
            obs, rewards, done, info = self.env.step(action)
            self.env.render()
            if done:
                obs = self.env.reset()
