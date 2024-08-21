import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np


# 定义Actor网络
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

# 定义Critic网络
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

# 定义Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, device):
        self.buffer = deque(maxlen=max_size)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

    def add(self, state, action, next_state, reward, not_done):
        self.buffer.append((state, action, next_state, reward, not_done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, next_state, reward, not_done = zip(*batch)
        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(self.device)
        not_done = torch.FloatTensor(np.array(not_done)).unsqueeze(1).to(self.device)
        return state, action, next_state, reward, not_done

    def __len__(self):
        return len(self.buffer)

# 定义SAC算法
class SAC:
    def __init__(self, state_dim, action_dim, max_action, device):
        self.device = device

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

        self.target_critic1 = Critic(state_dim, action_dim).to(self.device)
        self.target_critic2 = Critic(state_dim, action_dim).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.replay_buffer = ReplayBuffer(max_size=100000, state_dim=state_dim, action_dim=action_dim, device=self.device)
        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.005
        self.alpha = 0.2

    def select_action(self, state):
        with torch.no_grad():
            return self.actor(state)

    def update_parameters(self, batch_size):
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
# 示例使用
# if __name__ == "__main__":
#     state_dim = 9341  # 输入维度
#     action_dim = 10   # 输出维度（假设为10）
#     max_action = 1.0  # 假设动作的最大值为1.0
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     sac = SAC(state_dim, action_dim, max_action, device)
#
#     # 生成一个随机的状态张量
#     state = torch.randn(1, state_dim).to(device)
#
#     # 打印状态的形状
#     print(f"State shape: {state.shape}")
#
#     # 选择动作
#     action = sac.select_action(state)
#     print(f"Selected action: {action}")
#
#     # 添加到Replay Buffer并进行训练
#     next_state = torch.randn(1, state_dim).to(device)
#     reward = np.random.rand()
#     not_done = 1.0
#
#     sac.replay_buffer.add(state.cpu().numpy(), action.cpu().numpy(), next_state.cpu().numpy(), reward, not_done)

    # 训练SAC
    # sac.train(batch_size=256)
