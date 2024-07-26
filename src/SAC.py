import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from env import Env  # 确保你的环境文件名和类名正确
from gymnasium.spaces import MultiBinary, MultiDiscrete, Box, Dict


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        q = torch.relu(self.l1(torch.cat([state, action], 1)))
        q = torch.relu(self.l2(q))
        return self.l3(q)


class SAC:
    def __init__(self, state_dim, action_dim, max_action, env):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

        self.target_critic1 = Critic(state_dim, action_dim).to(device)
        self.target_critic2 = Critic(state_dim, action_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # 确保 observation_space 和 action_space 的 dtype 是正确的
        if isinstance(env.observation_space, Box):
            env.observation_space.dtype = np.float32
        if isinstance(env.action_space, Box):
            env.action_space.dtype = np.float32

        # 展平 observation_space
        if isinstance(env.observation_space, Dict):
            low = []
            high = []
            for space in env.observation_space.spaces.values():
                if isinstance(space, Box):
                    low.append(space.low.flatten())
                    high.append(space.high.flatten())
                elif isinstance(space, MultiBinary):
                    low.append(np.zeros(space.n, dtype=np.float32))
                    high.append(np.ones(space.n, dtype=np.float32))
                elif isinstance(space, MultiDiscrete):
                    low.append(np.zeros(space.nvec.shape, dtype=np.float32).flatten())
                    high.append((space.nvec - 1).astype(np.float32).flatten())
                else:
                    raise NotImplementedError(f"Unsupported space type: {type(space)}")
            low = np.concatenate([x.flatten() for x in low])
            high = np.concatenate([x.flatten() for x in high])
            flat_obs_space = Box(
                low=low,
                high=high,
                dtype=np.float32
            )
        else:
            flat_obs_space = env.observation_space

        self.replay_buffer = ReplayBuffer(
            buffer_size=100000,  # 调整为较小的值，例如 100000
            observation_space=flat_obs_space,
            action_space=env.action_space,
            device=device,
            optimize_memory_usage=False,
            handle_timeout_termination=False  # 确保与 optimize_memory_usage 兼容
        )
        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.005
        self.alpha = 0.2

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, batch_size=256):
        state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action = self.actor(next_state)
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = reward + not_done * self.discount * torch.min(target_q1, target_q2)

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        actor_loss = -self.critic1(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.critic1.state_dict(), filename + "_critic1")
        torch.save(self.critic2.state_dict(), filename + "_critic2")
        torch.save(self.target_critic1.state_dict(), filename + "_target_critic1")
        torch.save(self.target_critic2.state_dict(), filename + "_target_critic2")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.critic1.load_state_dict(torch.load(filename + "_critic1"))
        self.critic2.load_state_dict(torch.load(filename + "_critic2"))
        self.target_critic1.load_state_dict(torch.load(filename + "_target_critic1"))
        self.target_critic2.load_state_dict(torch.load(filename + "_target_critic2"))


if __name__ == "__main__":
    env = Env()  # 确保你的环境类名和实例化方式正确
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 输出设备信息
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
    else:
        print("CUDA is not available. Using CPU.")
    print(f"Device: {device}")

    # 获取 observation_space 和 action_space 的具体形状
    obs_space = env.observation_space
    act_space = env.action_space

    # 计算 state_dim 和 action_dim
    if isinstance(obs_space, Dict):
        state_dim = sum([np.prod(space.shape) for space in obs_space.spaces.values()])
    else:
        state_dim = np.prod(obs_space.shape)

    if isinstance(act_space, MultiBinary):
        action_dim = act_space.n
    else:
        action_dim = np.prod(act_space.shape)

    max_action = 1  # 对于 MultiBinary 动作空间，最大值为1

    sac = SAC(state_dim, action_dim, max_action, env)

    num_episodes = 1000
    max_timesteps = 200
    batch_size = 256

    for episode in range(num_episodes):
        state_dict, _ = env.reset()
        try:
            state = np.concatenate(
                [state_dict[key].flatten() for key in state_dict if hasattr(state_dict[key], 'flatten')])
        except Exception as e:
            print(f"Error in flattening state_dict: {e}")
            for key in state_dict:
                print(f"Key: {key}, Value: {state_dict[key]}, Type: {type(state_dict[key])}")
            raise e

        episode_reward = 0

        for t in range(max_timesteps):
            action = sac.select_action(state)
            action = action[:action_dim]  # 确保动作维度与动作空间匹配
            next_state_dict, reward, done, _, _ = env.step(action)
            try:
                next_state = np.concatenate([next_state_dict[key].flatten() for key in next_state_dict if
                                             hasattr(next_state_dict[key], 'flatten')])
            except Exception as e:
                print(f"Error in flattening next_state_dict: {e}")
                for key in next_state_dict:
                    print(f"Key: {key}, Value: {next_state_dict[key]}, Type: {type(next_state_dict[key])}")
                raise e

            not_done = 1.0 if not done else 0.0

            sac.replay_buffer.add(state, action, next_state, reward, not_done)
            state = next_state
            episode_reward += reward

            if len(sac.replay_buffer) > batch_size:
                sac.train(batch_size)

            if done:
                break

        print(f"Episode {episode + 1}, Reward: {episode_reward}")

        if (episode + 1) % 100 == 0:
            sac.save(f"sac_checkpoint_{episode + 1}")

    env.close()
