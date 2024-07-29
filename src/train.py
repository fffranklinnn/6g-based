import torch
import numpy as np
from env import Env  # 假设 env.py 中的 Env 类已经导入
from sac import SAC
from utils import flatten_state


def main():
    env = Env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    state_dim = env.get_observation_shape()[0]
    action_dim = env.action_space.numel()  # 确保获取正确的动作维度
    max_action = 1  # 对于 MultiBinary 动作空间，最大值为1

    sac = SAC(state_dim, action_dim, max_action, device)  # 传递 device 参数

    num_episodes = 100
    max_timesteps = 60
    batch_size = 256

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = flatten_state(state).to(device)  # 确保状态在正确的设备上并平整化

        episode_reward = 0

        for t in range(max_timesteps):
            action = sac.select_action(state)
            action = torch.clamp(action, 0, 1)  # 确保动作在有效范围内

            # 确保action是一个numpy数组并且形状正确
            action = action.cpu().numpy()
            action = action.reshape((env.NUM_SATELLITES, env.NUM_GROUND_USER))
            print(f"Action shape after reshape: {action.shape}")

            next_state, reward, done, _ = env.step(action)  # 保持动作为numpy类型
            next_state = flatten_state(next_state).to(device)  # 平整并移至正确的设备

            not_done = 1.0 if not done else 0.0

            sac.replay_buffer.add(state.cpu().numpy(), action, next_state.cpu().numpy(), reward, not_done)  # 处理设备转移问题
            state = next_state
            episode_reward += reward

            if len(sac.replay_buffer) > batch_size:
                sac.train(batch_size)

            if done:
                break

        print(f"Episode {episode + 1}, Reward: {episode_reward}")

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1} completed. Saving model checkpoint.")
            sac.save(f"sac_checkpoint_{episode + 1}")

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean([env.step(sac.select_action(flatten_state(env.reset()[0]).to(device)))[1] for _ in range(10)])
            print(f"Episode {episode + 1}, Average Reward over 10 episodes: {avg_reward}")

    env.close()


if __name__ == "__main__":
    main()
