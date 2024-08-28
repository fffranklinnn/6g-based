import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import torch.nn.functional as F
# --------------------------------------- #
# 经验回放池
# --------------------------------------- #
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

# -------------------------------------- #
# 构造深度学习网络，输入状态s，得到各个动作的reward
# -------------------------------------- #

class Net(nn.Module):
    # 构造只有一个隐含层的网络
    def __init__(self, state_dim, num_users=10, num_satellites=301):
        super(Net, self).__init__()
        self.num_users = num_users
        self.num_satellites = num_satellites
        action_dim = num_users * num_satellites
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, action_dim)

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        a = self.l3(a)
        a = a.view(-1, self.num_users, self.num_satellites)
        a = torch.softmax(a, dim=-1)
        return a
        # return self.max_action * torch.tanh(self.l3(a))

class target_q_net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(target_q_net, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, state, action):
        # 打印state和action的形状以帮助调试
        # print(f"State shape: {state.shape}")
        # print(f"Action shape: {action.shape}")

        # 确保state和action可以被正确拼接
        state = torch.squeeze(state, dim=1)
        action = torch.squeeze(action, dim=1)
        q_input = torch.cat((state, action), dim=1)
        # q = torch.relu(self.l1(torch.cat([state, action], dim=1)))
        q = self.l1(q_input)
        q = torch.relu(self.l2(q))
        return self.l3(q)
# -------------------------------------- #
# 构造深度强化学习模型
# -------------------------------------- #
class DQN:
    # （1）初始化
    def __init__(self, state_dim,action_dim, device):
        # 属性分配

        self.discount = 0.90
        self.epsilon = 1# 贪婪策略，有1-epsilon的概率探索
        self.target_update = 20   # 目标网络的参数的更新频率
        self.device = device  # 在GPU计算
        # 计数器，记录迭代次数
        self.count = 0

        # 构建2个神经网络，相同的结构，不同的参数
        self.q_net = Net(state_dim).to(self.device)
        # 实例化目标网络
        self.target_q_net = target_q_net(state_dim,action_dim).to(self.device)

        # 优化器，更新训练网络的参数
        self.target_q_net_optimizer = torch.optim.Adam(self.target_q_net.parameters(), lr=3e-6)
        self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=3e-6)
        self.replay_buffer = ReplayBuffer(max_size=100000, state_dim=state_dim, action_dim=action_dim,
                                      device=self.device)
    # （2）动作选择
    def take_action(self, state, action_dim):
        state = state.to(self.device)
        state = state.unsqueeze(0)
        # 如果小于该值就取最大的值对应的索引
        if np.random.random() < self.epsilon:  # 0-1
            # 前向传播获取该状态对应的动作的reward
            action = self.q_net(state).cpu().data.numpy().flatten()

            # 获取reward最大值对应的动作索引
            # action = actions_value.argmax().item()  # int
        # 如果大于该值就随机探索
        else:
            # 随机选择一个动作
            action = torch.randint(0, 2, (action_dim,))
            action = action.data.numpy().flatten()
        with torch.no_grad():
            return action

    # （3）网络训练
    def update(self, batch_size):  # 传入经验池中的batch个样本
        if len(self.replay_buffer) < batch_size:
            return
        state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)

        # 归一化状态和下一状态（避免不必要的数据传输）
        state_np = state.cpu().numpy()
        next_state_np = next_state.cpu().numpy()
        state = torch.FloatTensor(state_np).to(self.device)
        next_state = torch.FloatTensor(next_state_np).to(self.device)


        q_values = self.target_q_net(state,action)
        with torch.no_grad():
            next_action = self.q_net(next_state)
            # 假设 next_action 是三维的：批次大小 x 用户数 x 卫星数
            next_action_flattened = next_action.view(next_action.size(0), -1)  # 展平为二维张量
            max_next_q_values = self.target_q_net(next_state,next_action_flattened)
            # 目标网络输出的当前状态的q(state_value)：即时奖励+折扣因子*下个时刻的最大回报
            q_targets = reward.unsqueeze(1) + not_done.unsqueeze(1) * self.discount * max_next_q_values

        # 目标网络输出的当前状态的q(state_value)：即时奖励+折扣因子*下个时刻的最大回报
        #print("reward shape:", reward.shape)
        #print("not_done shape:", not_done.shape)
        #print("max_next_q_values shape:", max_next_q_values.shape)

        #q_targets = reward.unsqueeze(1) + not_done.unsqueeze(1) * self.discount * max_next_q_values
        # 目标网络和训练网络之间的均方误差损失
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        self.target_q_net_optimizer.zero_grad()
        # 反向传播参数更新
        dqn_loss.backward()
        # 对训练网络更新
        self.target_q_net_optimizer.step()
        # 在计算 actor_loss 之前，调整动作张量的维度
        predicted_action = self.q_net(state)  # 这可能是三维的
        predicted_action_flattened = predicted_action.view(predicted_action.size(0), -1)  # 展平为二维张量
        q_net_loss = -self.target_q_net(state, predicted_action_flattened).mean()

        self.q_net_optimizer.zero_grad()
        q_net_loss.backward()
        self.q_net_optimizer.step()  # 更新参数
        '''
        for name1,param1 in self.target_q_net.named_parameters():
            if param1.grad is not None:
                print(str(name1)+' has grad')
            else:
                print(str(name1)+' none grad')
        '''
        # 在一段时间后更新目标网络的参数
        if self.count % self.target_update == 0:
            # 将目标网络的参数替换成训练网络的参数
            self.target_q_net.load_state_dict(
                self.target_q_net.state_dict())

        self.count += 1