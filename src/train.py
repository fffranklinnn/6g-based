import torch
import numpy as np
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

    sac = SAC(flattened_state_dim, action_dim, max_action, device)

    num_episodes = 100
    max_timesteps = 60
    batch_size = 256

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = flatten_state(state).to(device)

        episode_reward = 0

        for t in range(max_timesteps):
            action = sac.select_action(state)
            action = torch.clamp(action, 0, 1)
            action_numpy = action.cpu().numpy().reshape((env.NUM_SATELLITES, env.NUM_GROUND_USER))

            try:
                next_state, reward, done, _ = env.step(action_numpy)  # 确保传递的是正确的numpy数组
            except Exception as e:
                print(f"Error during environment step: {e}")
                break

            next_state = flatten_state(next_state).to(device)
            not_done = 1.0 - float(done)

            sac.replay_buffer.add(state.cpu().numpy(), action_numpy, next_state.cpu().numpy(), reward, not_done)
            state = next_state
            episode_reward += reward

            if len(sac.replay_buffer) > batch_size:
                sac.update_parameters(batch_size)

            if done:
                break

        print(f"Episode {episode + 1}, Reward: {episode_reward}")

        if (episode + 1) % 10 == 0:
            sac.save(f"sac_checkpoint_{episode + 1}")

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean([env.step(sac.select_action(flatten_state(env.reset()[0]).to(device)))[1] for _ in range(10)])
            print(f"Episode {episode + 1}, Average Reward over 10 episodes: {avg_reward}")

    env.close()


if __name__ == "__main__":
    main()
