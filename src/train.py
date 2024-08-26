# import random
import torch
import numpy as np
from matplotlib import pyplot as plt

from env import Env  # 假设 env.py 中的 Env 类已经导入
from sac import SAC
from utils import flatten_state


def main():
    env = Env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    raw_state_dim = env._calculate_observation_shape()[0]
    flattened_state_dim = flatten_state(torch.zeros(raw_state_dim)).numel()

    action_dim = env.action_space.numel()
    max_action = 1  # 确认这是否适合你的环境

    sac = SAC(flattened_state_dim, action_dim, max_action, device, env.NUM_SATELLITES, env.NUM_GROUND_USER)

    num_episodes = 10
    max_timesteps = 60
    batch_size = 256

    def adjust_action(action, num_satellites, num_users):
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action)
        action_2d = action.view(num_satellites, num_users)

        # 确保每个用户只连接一个卫星，每个卫星只服务一个用户
        for satellite_idx in range(num_satellites):
            selected_user_idx = torch.argmax(action_2d[satellite_idx])
            for user_idx in range(num_users):
                if user_idx != selected_user_idx:
                    action_2d[satellite_idx][user_idx] = 0

        for user_idx in range(num_users):
            # 获取当前列的索引
            col_indices = torch.nonzero(action_2d[:, user_idx]).view(-1)

            # 如果当前列有多个元素为1，随机选择一个索引，将其他元素设为0
            if col_indices.size(0) > 1:
                selected_idx = torch.randint(0, col_indices.size(0), (1,))
                for idx in range(col_indices.size(0)):
                    if idx != selected_idx:
                        action_2d[col_indices[idx], user_idx] = 0
        # 将调整后的二维数组重新展平为一维数组作为最终动作
        adjusted_action = action_2d.view(-1)

        return adjusted_action

    episode_rewards = []  # 用于存储每个 episode 的奖励
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = flatten_state(state).to(device)
        # print(state)
        episode_reward = 0

        for t in range(max_timesteps):
            action = sac.select_action(state)
            action_tensor = torch.tensor(action).to(device)  # 确保 action 是 PyTorch 张量
            adjusted_action = adjust_action(action_tensor, env.NUM_SATELLITES, env.NUM_GROUND_USER)
            #print(action)
            #action = torch.clamp(action, 0, 1)
            #print(action)
            #action_numpy = action.cpu().numpy().reshape((env.NUM_SATELLITES, env.NUM_GROUND_USER))

            try:
                next_state, reward, done, _ = env.step(adjusted_action.cpu().numpy())  # 确保传递的是正确的numpy数组
            except Exception as e:
                print(f"Error during environment step: {e}")
                break

            next_state = flatten_state(next_state).to(device)
            not_done = 1.0 - float(done)

            sac.replay_buffer.add(state.cpu().numpy(), adjusted_action.cpu().numpy(), next_state.cpu().numpy(), reward, not_done)
            state = next_state
            episode_reward += reward

            if len(sac.replay_buffer) > batch_size:
               sac.update_parameters(batch_size)

            if done:
                break
        episode_rewards.append(episode_reward.cpu().numpy())  # 将张量移动到 CPU 上并转换为 NumPy 数组，然后记录当前 episode 的奖励
        print(f"Episode {episode + 1}, Reward: {episode_reward}")

        if (episode + 1) % 10 == 0:
            sac.save(f"sac_checkpoint_{episode + 1}")

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean([env.step(sac.select_action(flatten_state(env.reset()[0]).to(device)))[1] for _ in range(10)])
            print(f"Episode {episode + 1}, Average Reward over 10 episodes: {avg_reward}")

    plt.figure()
    plt.plot(range(1, num_episodes + 1), episode_rewards, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.grid()
    plt.show()
    env.close()


if __name__ == "__main__":
    main()