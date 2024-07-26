import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from env import Env  # 确保你的环境文件名和类名正确

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
        q = torch.relu(self.l1(torch.cat([state, action], dim=1)))
        q = torch.relu(self.l2(q))
        return self.l3(q)

class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.buffer = deque(maxlen=max_size)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def add(self, state, action, next_state, reward, not_done):
        self.buffer.append((state, action, next_state, reward, not_done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, next_state, reward, not_done = zip(*batch)
        state = torch.FloatTensor(np.array(state)).to(device)
        action = torch.FloatTensor(np.array(action)).to(device)
        next_state = torch.FloatTensor(np.array(next_state)).to(device)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(device)
        not_done = torch.FloatTensor(np.array(not_done)).unsqueeze(1).to(device)
        return state, action, next_state, reward, not_done

    def __len__(self):
        return len(self.buffer)

class SAC:
    def __init__(self, state_dim, action_dim, max_action):
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

        self.replay_buffer = ReplayBuffer(max_size=100000, state_dim=state_dim, action_dim=action_dim)
        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.005
        self.alpha = 0.2

    def select_action(self, state):
        with torch.no_grad():
            return self.actor(state).cpu()

    def train(self, batch_size=256):
        if len(self.replay_buffer) < batch_size:
            return

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

def flatten_state(state):
    return state.flatten()

if __name__ == "__main__":
    env = Env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = env.get_observation_shape()[0]
    action_dim = env.action_space.numel()  # 确保获取正确的动作维度
    max_action = 1  # 对于 MultiBinary 动作空间，最大值为1

    sac = SAC(state_dim, action_dim, max_action)

    num_episodes = 1000
    max_timesteps = 200
    batch_size = 256

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(flatten_state(state)).to(device)  # 确保状态在正确的设备上并平整化

        episode_reward = 0

        for t in range(max_timesteps):
            action = sac.select_action(state)
            action = torch.clamp(action, 0, 1).to(device)  # 确保动作在有效范围内并保持为tensor
            next_state, reward, done, _ = env.step(action)  # 直接传递tensor
            next_state = torch.FloatTensor(flatten_state(next_state)).to(device)  # 平整并移至正确的设备

            not_done = 1.0 if not done else 0.0

            sac.replay_buffer.add(state.cpu().numpy(), action.cpu().numpy(), next_state.cpu().numpy(), reward, not_done)  # 处理设备转移问题
            state = next_state
            episode_reward += reward

            sac.train(batch_size)

            if done:
                break

        print(f"Episode {episode + 1}, Reward: {episode_reward}")

        if (episode + 1) % 100 == 0:
            sac.save(f"sac_checkpoint_{episode + 1}")

    env.close()
