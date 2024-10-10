# import random
import torch
import numpy as np
import os
import csv
from matplotlib import pyplot as plt

from env import Env  # 假设 env.py 中的 Env 类已经导入
from DQN import DQN
from utils import flatten_state
import Normalizer


def main():
    env = Env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    raw_state_dim = env._calculate_observation_shape()[0]
    flattened_state_dim = flatten_state(torch.zeros(raw_state_dim)).numel()

    action_dim = env.action_space.numel()
    max_action = 1  # 确认这是否适合你的环境

    dqn = DQN(flattened_state_dim, action_dim, device)

    normalizer = Normalizer.ComplexNormalizer(env.NUM_SATELLITES, env.NUM_GROUND_USER)
    num_episodes = 200
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

    episode_rewards_dqn = []  # 用于存储每个 episode 的奖励
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = flatten_state(state).to(device)
        # 假设state是一个PyTorch张量
        if isinstance(state, torch.Tensor):
            state_np = state.cpu().numpy()  # 直接转换为NumPy数组
        else:
            state_np = state  # state已经是NumPy数组或其他类型，不需要转换

        # 现在可以安全地调用update方法
        normalizer.update(state_np)

        normalizer_state = normalizer.normalize(state_np)
        normalized_state_tensor = torch.tensor(normalizer_state).float().to(device)

        state = normalized_state_tensor  # 此时的state为PyTorch张量
        episode_reward_dqn = 0

        for t in range(max_timesteps):
            action_dqn = dqn.take_action(state, action_dim)
            action_tensor = torch.tensor(action_dqn).to(device)  # 确保 action 是 PyTorch 张量
            action_dqn = adjust_action(action_tensor, env.NUM_SATELLITES, env.NUM_GROUND_USER)
            try:
                next_state, reward_dqn, done, _ = env.step(action_dqn.cpu().numpy())  # 使用转换后的action
            except Exception as e:
                print(f"Error during environment step: {e}")
                break

            next_state = flatten_state(next_state).to(device)
            if isinstance(next_state, torch.Tensor):
                next_state_np = next_state.cpu().numpy()  # 转换为NumPy数组
            else:
                next_state_np = next_state  # next_state已经是NumPy数组或其他类型，不需要转换

            # 现在可以安全地调用update方法
            normalizer.update(next_state_np)

            normalizer_state = normalizer.normalize(next_state_np)
            normalized_state_tensor = torch.tensor(normalizer_state).float().to(device)

            next_state = normalized_state_tensor
            not_done = 1.0 - float(done)

            # 现在state, action, next_state都是PyTorch张量，需要转换为NumPy数组以存储
            dqn.replay_buffer.add(state.cpu().numpy(), action_tensor.cpu().numpy(), next_state.cpu().numpy(),
                                  reward_dqn,
                                  not_done)

            state = next_state  # 更新state为下一状态（PyTorch张量）
            episode_reward_dqn += reward_dqn

            if len(dqn.replay_buffer) > batch_size:
                dqn.update(batch_size)

            if done:
                break
        episode_rewards_dqn.append(episode_reward_dqn.cpu())  # 记录当前 episode 的奖励
        print(f"Episode {episode + 1}, Reward: {episode_reward_dqn}")

    os.makedirs('./data', exist_ok=True)
    plt.figure()
    plt.plot(range(1, num_episodes + 1), episode_rewards_dqn, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.grid()
    plt.show()

    with open("data_sac.csv", "w", newline='') as fo:
        header = ["Episode", "Reward"]
        writer = csv.DictWriter(fo, fieldnames=header)
        writer.writeheader()
        for episode, reward in zip(range(1, num_episodes + 1), episode_rewards_dqn):
            writer.writerow({"Episode": episode, "Reward": reward})

    # fo = open("data_sac.csv", "w", newline='')
    # header = ["Episode", "Reward"]
    # writer = csv.DictWriter(fo, fieldnames=header)
    # writer.writeheader()
    np.savez_compressed('./data/dqn_dataset', x=np.arange(num_episodes), episode_rewards=episode_rewards_dqn)
    env.close()

#.git

if __name__ == "__main__":
    main()
