import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from Normalizer import ComplexNormalizer  # 导入 ComplexNormalizer 类


# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, action_dim)
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
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 1)

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
        state = torch.as_tensor(state, device=self.device)
        action = torch.as_tensor(action, device=self.device)
        next_state = torch.as_tensor(next_state, device=self.device)
        reward = torch.as_tensor(reward, device=self.device, dtype=torch.float32)
        not_done = torch.as_tensor(not_done, device=self.device, dtype=torch.float32)
        self.buffer.append((state, action, next_state, reward, not_done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, next_state, reward, not_done = zip(*batch)

        state = torch.stack(state, dim=0).to(self.device)
        action = torch.stack(action, dim=0).to(self.device)
        next_state = torch.stack(next_state, dim=0).to(self.device)
        reward = torch.stack(reward, dim=0).to(self.device)
        not_done = torch.stack(not_done, dim=0).to(self.device)

        return state, action, next_state, reward, not_done

    def __len__(self):
        return len(self.buffer)


# 定义SAC算法
class SAC:
    def __init__(self, state_dim, action_dim, max_action, device, num_satellites, num_ground_user):
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

        self.replay_buffer = ReplayBuffer(max_size=100000, state_dim=state_dim, action_dim=action_dim,
                                          device=self.device)
        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.005
        self.alpha = 0.2

        self.normalizer = ComplexNormalizer(num_satellites, num_ground_user)  # 初始化归一化器

    def select_action(self, state):
        state = self.normalizer.normalize(state.cpu().numpy())  # 归一化状态
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.actor(state).cpu().data.numpy().flatten()

    def update_parameters(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)

        state = self.normalizer.normalize(state.cpu().numpy())  # 归一化状态
        next_state = self.normalizer.normalize(next_state.cpu().numpy())  # 归一化下一状态

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)

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

    def interact_with_environment(self, env, num_steps):
        state = env.reset()
        for _ in range(num_steps):
            action = self.select_action(state)
            next_state, reward, done, _ = env.step(action)
            self.normalizer.update(state)  # 更新归一化器
            self.replay_buffer.add(state, action, next_state, reward, float(not done))
            state = next_state
            if done:
                state = env.reset()
