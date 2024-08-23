import torch
import numpy as np
import matplotlib.pyplot as plt
from env import Env  # 假设 env.py 中的 Env 类已经导入
from sac import SAC
from utils import flatten_state


def plot_rewards(rewards, avg_rewards, save_path='rewards_plot.png'):
    plt.figure(figsize=(12, 8))
    plt.plot(rewards, label='Episode Reward')
    plt.plot(avg_rewards, label='Average Reward (over 10 episodes)', linestyle='--')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def main():
    env = Env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    raw_state_dim = env._calculate_observation_shape()[0]
    flattened_state_dim = flatten_state(torch.zeros(raw_state_dim)).numel()

    action_dim = env.action_space.numel()
    max_action = 1  # 确认这是否适合你的环境

    sac = SAC(flattened_state_dim, action_dim, max_action, device)

    # 尝试加载之前保存的模型
    try:
        sac.load("sac_checkpoint")
        print("模型加载成功")
    except FileNotFoundError:
        print("没有找到保存的模型，开始新的训练")

    num_episodes = 100
    max_timesteps = 60
    batch_size = 256
    save_interval = 10  # 每隔多少个 episode 保存一次模型

    episode_rewards = []
    avg_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = flatten_state(state).to(device)

        episode_reward = 0

        for t in range(max_timesteps):
            action = sac.select_action(state)
            action = torch.clamp(action, 0, 1)

            try:
                next_state, reward, done, _ = env.step(action)
            except Exception as e:
                print(f"Error during environment step: {e}")
                break

            next_state = flatten_state(next_state).to(device)
            not_done = 1.0 - float(done)

            sac.replay_buffer.add(state.cpu().numpy(), action, next_state.cpu().numpy(), reward, not_done)
            state = next_state
            episode_reward += reward

            if len(sac.replay_buffer) > batch_size:
                sac.update_parameters(batch_size)

            if done:
                break

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}, Reward: {episode_reward}")

        # 定期保存模型
        if (episode + 1) % save_interval == 0:
            sac.save("sac_checkpoint")
            print(f"模型已保存: Episode {episode + 1}")

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(
                [env.step(sac.select_action(flatten_state(env.reset()[0]).to(device)))[1] for _ in range(10)])
            avg_rewards.append(avg_reward)
            print(f"Episode {episode + 1}, Average Reward over 10 episodes: {avg_reward}")

    # 训练结束后保存最终模型
    sac.save("sac_final")
    print("最终模型已保存")

    # 绘制奖励曲线
    plot_rewards(episode_rewards, avg_rewards)

    env.close()


if __name__ == "__main__":
    main()
