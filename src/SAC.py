from stable_baselines3 import SAC
import logging

class RLAlgorithm:
    def __init__(self, env, model_path="sac_env"):
        """
        初始化RL算法类。

        参数:
        env -- 强化学习环境
        model_path -- 模型保存路径
        """
        self.env = env
        self.model_path = model_path
        self.model = None
        logging.basicConfig(level=logging.INFO)

    def create_model(self, policy="MlpPolicy", **kwargs):
        """
        创建并返回一个SAC模型。

        参数:
        policy -- 策略网络类型
        kwargs -- 其他SAC模型参数
        """
        self.model = SAC(policy, self.env, verbose=1, **kwargs)
        logging.info("Model created with policy: %s", policy)
        return self.model

    def train_model(self, total_timesteps=10000):
        """
        训练模型。

        参数:
        total_timesteps -- 训练的总步数
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        logging.info("Starting training for %d timesteps", total_timesteps)
        self.model.learn(total_timesteps=total_timesteps)
        logging.info("Training completed")
        return self.model

    def save_model(self):
        """
        保存模型到指定路径。
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        self.model.save(self.model_path)
        logging.info("Model saved to %s", self.model_path)

    def load_model(self):
        """
        从指定路径加载模型。
        """
        self.model = SAC.load(self.model_path, env=self.env)
        logging.info("Model loaded from %s", self.model_path)
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
        logging.info("Starting evaluation for %d steps", num_steps)
        for i in range(num_steps):
            action, _states = self.model.predict(obs)
            obs, rewards, done, info = self.env.step(action)
            self.env.render()
            if done:
                logging.info("Environment reset at step %d", i)
                obs = self.env.reset()
        logging.info("Evaluation completed")

# 示例用法
if __name__ == "__main__":
    import gym
    env = gym.make('Pendulum-v1')

    rl_algorithm = RLAlgorithm(env)
    rl_algorithm.create_model(learning_rate=3e-4, buffer_size=1000000, learning_starts=10000, batch_size=256, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1, ent_coef='auto', target_update_interval=1, target_entropy='auto')
    rl_algorithm.train_model(total_timesteps=100000)
    rl_algorithm.save_model()
    rl_algorithm.load_model()
    rl_algorithm.evaluate_model(num_steps=1000)
