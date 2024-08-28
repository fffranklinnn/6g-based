# import numpy as np
import pandas as pd
import torch
import os
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_action_matrix(action_matrix):
    plt.figure(figsize=(12, 5))  # 设置图像大小
    sns.heatmap(action_matrix, cmap="YlGnBu", cbar=True, annot=False)
    plt.title("Action Matrix Heatmap")
    plt.xlabel("Ground User")
    plt.ylabel("Satellite")
    plt.show()


class Env:
    def __init__(self):
        super(Env, self).__init__()
        # 定义常量参数
        self.NUM_SATELLITES = 301  # 卫星数量
        self.NUM_GROUND_USER = 10  # 地面用户数量
        self.TOTAL_TIME = 3000  # 总模拟时间，单位：秒
        self.NUM_TIME_SLOTS = 61  # 时间段的划分数量
        self.TIME_SLOT_DURATION = 50  # 每个时间段的持续时间
        self.communication_frequency = torch.tensor(18.5e9)  # 通信频率为18.5GHz
        self.total_bandwidth = 250e6  # 总带宽为250MHz
        self.noise_temperature = 213.15  # 系统的噪声温度为213.15开尔文
        self.Polarization_isolation_factor = 12  # 单位dB
        self.receive_benefit_ground = 15.4  # 单位dB
        self.EIRP = 73.1  # 单位:dBm
        self.k = 1.380649e-23  # 单位:J/K
        self.radius_earth = 6731e3  # 单位:m
        self.EIRP_watts = 10 ** ((self.EIRP - 30) / 10)  # 将 EIRP 从 dBm 转换为瓦特
        self.noise_power = self.k * self.noise_temperature * self.total_bandwidth  # 噪声功率计算
        self.angle_threshold = 15  # 单位：度
        self.w1 = 5e-6  # 切换次数的权重
        self.w2 = 1e-7  # 用户传输速率的权重

        # 定义动作空间和观察空间
        self.action_space = torch.zeros(self.NUM_SATELLITES * self.NUM_GROUND_USER, dtype=torch.int)
        self.coverage_space = torch.zeros((self.NUM_TIME_SLOTS, self.NUM_SATELLITES, self.NUM_GROUND_USER, 2),
                                          dtype=torch.int)
        self.previous_access_strategy_space = torch.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER), dtype=torch.int)
        self.switch_count_space = torch.zeros(self.NUM_GROUND_USER, dtype=torch.int)
        self.elevation_angle_space = torch.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER), dtype=torch.float32)
        self.altitude_space = torch.zeros(self.NUM_SATELLITES, dtype=torch.float32)
        self.observation_space = {
            "coverage": self.coverage_space,
            "previous_access_strategy": self.previous_access_strategy_space,
            "switch_count": self.switch_count_space,
            "elevation_angles": self.elevation_angle_space,
            "altitudes": self.altitude_space
        }
        self.action_dim = self.NUM_SATELLITES * self.NUM_GROUND_USER
        self.state_dim = (
                self.NUM_SATELLITES * self.NUM_GROUND_USER * 2 +  # coverage_space
                self.NUM_SATELLITES * self.NUM_GROUND_USER +  # previous_access_strategy_space
                self.NUM_GROUND_USER +  # switch_count_space
                self.NUM_SATELLITES * self.NUM_GROUND_USER +  # elevation_angle_space
                self.NUM_SATELLITES  # altitude_space
        )

        # 初始化卫星和用户的位置
        self.satellite_heights = self.initialize_altitude()
        self.eval_angle = self.initialize_angle()

        # 初始化覆盖指示变量和接入决策变量
        self.coverage_indicator = self.initialize_coverage()
        self.access_decision = torch.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER), dtype=torch.int)  # 仅存储当前时隙的接入决策

        self.current_time_step = 0
        self.switch_count = torch.zeros(self.NUM_GROUND_USER, dtype=torch.int)

        # 初始化用户需求传输速率矩阵
        self.user_rate = torch.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER), dtype=torch.float32)

        # 初始化信道容量矩阵
        self.channel_capacity = torch.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER), dtype=torch.float32)

        # 初始化用户需求速率
        self.user_demand_rate = torch.empty((self.NUM_GROUND_USER, self.NUM_TIME_SLOTS), dtype=torch.float32)
        self.user_demand_rate.uniform_(1e9, 10e9)

        # 使用 GPU 加速
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def to(self, device):
        self.coverage_indicator = self.coverage_indicator.to(device)
        self.access_decision = self.access_decision.to(device)
        self.switch_count = self.switch_count.to(device)
        self.user_rate = self.user_rate.to(device)
        self.channel_capacity = self.channel_capacity.to(device)
        self.user_demand_rate = self.user_demand_rate.to(device)
        self.satellite_heights = self.satellite_heights.to(device)
        self.eval_angle = self.eval_angle.to(device)
        self.previous_access_strategy_space = self.previous_access_strategy_space.to(self.device)
        # print(f"Moved to device: {device}")

    def initialize_angle(self):
        # 从CSV文件读取地面用户的仰角数据
        df = pd.read_csv('ev_data.csv')
        eval_angle = torch.zeros((self.NUM_TIME_SLOTS, self.NUM_GROUND_USER, self.NUM_SATELLITES), dtype=torch.float32)

        # 填充 eval_angle 数组
        for time_slot in range(self.NUM_TIME_SLOTS):
            for i in range(self.NUM_SATELLITES):
                for j in range(self.NUM_GROUND_USER):
                    eval_angle[time_slot, j, i] = df.iloc[i * self.NUM_GROUND_USER + j, time_slot]

        # print(f"Initialized eval_angle with shape: {eval_angle.shape}")  # [TIME_SLOTS,NUM_GROUND_USER,NUM_SATELLITES]
        return eval_angle

    def initialize_altitude(self):
        # 检查CSV文件是否存在
        csv_file = 'alt_data.csv'
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"{csv_file} does not exist.")
        # 从CSV文件读取卫星的高度数据
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            raise IOError(f"Error reading {csv_file}: {e}")

        # print(f"alt DataFrame shape: {df.shape}")

        # 检查 DataFrame 的形状是否符合预期
        if df.shape[0] < self.NUM_SATELLITES or df.shape[1] < self.NUM_TIME_SLOTS:
            raise ValueError(
                f"DataFrame shape {df.shape} is not sufficient for the given NUM_SATELLITES and NUM_TIME_SLOTS")

        sat_heights = torch.zeros((self.NUM_TIME_SLOTS, self.NUM_SATELLITES), dtype=torch.float32)

        # 填充 sat_heights 数组
        for time_slot in range(self.NUM_TIME_SLOTS):
            for i in range(self.NUM_SATELLITES):
                try:
                    # 确保从 DataFrame 中读取的数据是数值类型
                    value = float(df.iloc[i, time_slot])
                    sat_heights[time_slot, i] = value
                except ValueError:
                    raise ValueError(f"Non-numeric data found at row {1 + i}, column {time_slot}")
                except IndexError:
                    raise IndexError(f"Index out of bounds: row {1 + i}, column {time_slot}")

        # print(f"Initialized sat_heights with shape: {sat_heights.shape}")  # [NUM_TIME_SLOTS, NUM_SATELLITES]
        return sat_heights

    def initialize_coverage(self):
        csv_file = 'coverage_data.csv'
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"{csv_file} does not exist.")
        try:
            # 跳过第一行（时间戳行）
            df = pd.read_csv(csv_file, header=None, skiprows=1)
        except Exception as e:
            raise IOError(f"Error reading {csv_file}: {e}")
        # print(f"cov DataFrame shape: {df.shape}")

        if df.shape[0] < self.NUM_GROUND_USER * self.NUM_SATELLITES or df.shape[1] < self.NUM_TIME_SLOTS:
            raise ValueError(
                f"DataFrame shape {df.shape} is not sufficient for the given NUM_SATELLITES and NUM_TIME_SLOTS")

        # 从CSV文件读取覆盖数据
        coverage = torch.zeros((self.NUM_TIME_SLOTS, self.NUM_GROUND_USER, self.NUM_SATELLITES, 2), dtype=torch.float32)

        # 填充 coverage 数组
        for time_slot in range(self.NUM_TIME_SLOTS):
            for i in range(self.NUM_SATELLITES):
                for j in range(self.NUM_GROUND_USER):
                    # 获取覆盖字符串并解析成整数
                    coverage_str = df.iloc[j + i * self.NUM_GROUND_USER, time_slot]
                    beam_1, beam_2 = map(int, coverage_str.strip('()').split(','))
                    coverage[time_slot, j, i, 0] = beam_1
                    coverage[time_slot, j, i, 1] = beam_2

        # print(f"Initialized coverage with shape: {coverage.shape}")
        return coverage

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        # print(f"Starting step {self.current_time_step}/{self.NUM_TIME_SLOTS}")

        # 确保 action 是一个 torch.Tensor
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)  # 根据你的需要可能需要调整数据类型

        # 确保 action 的形状和数据类型正确
        try:
            action_matrix = action.view((self.NUM_SATELLITES, self.NUM_GROUND_USER)).to(self.device)
        except Exception as e:
            print(f"Error reshaping or moving action to device: {e}")
            raise

        if self.current_time_step >= self.NUM_TIME_SLOTS:
            print("Reached the end of time slots, terminating...")
            return self.terminate()

        self.access_decision = action_matrix
        # numpy_array = self.access_decision.cpu().numpy()

        if self.current_time_step > 0:
            self.update_switch_count(action_matrix)

        self.update_rates_and_capacity(action_matrix)
        reward = self.calculate_reward(action_matrix)
        # print(f"Reward at step {self.current_time_step}: {reward}")

        self.current_time_step += 1
        is_done = self.current_time_step >= self.NUM_TIME_SLOTS
        # print(f"End of step {self.current_time_step}, Termination status: {is_done}")

        observation = self.get_observation() if not is_done else torch.zeros(self._calculate_observation_shape(),
                                                                             device=self.device)
        # print(f"Observation for next step: {observation.shape}")

        information = {'current_time_step': self.current_time_step, 'switch_count': self.switch_count.clone()}
        # 在这里调用可视化函数
        # visualize_action_matrix(action_matrix.cpu().numpy())  # 确保数据在 CPU 上，并转换为 NumPy 数组
        return observation, reward, is_done, information

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[torch.Tensor, dict]:
        # 重置当前时间步
        self.current_time_step = 0

        # 重置覆盖指示变量和接入决策变量
        self.coverage_indicator = self.initialize_coverage()
        self.access_decision = torch.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER), dtype=torch.int,
                                           device=self.device)

        # 假设 self.access_decision 在此之前已经根据某种逻辑被赋值
        self.previous_access_strategy_space = self.access_decision.clone()

        # 重置切换次数
        self.switch_count = torch.zeros(self.NUM_GROUND_USER, dtype=torch.int, device=self.device)

        # 重置用户传输速率
        self.user_rate = torch.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER), dtype=torch.float32,
                                     device=self.device)

        # 重置信道容量矩阵
        self.channel_capacity = torch.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER), dtype=torch.float32,
                                            device=self.device)

        # 重置用户需求速率
        self.user_demand_rate = torch.empty((self.NUM_GROUND_USER, self.NUM_TIME_SLOTS), dtype=torch.float32,
                                            device=self.device)
        self.user_demand_rate.uniform_(1e12, 10e13)

        # 获取初始观察
        observation = self.get_observation()
        # print(f"Reset observation shape: {observation.shape}")

        return observation, {'current_time_step': self.current_time_step}

    def get_observation(self) -> torch.Tensor:
        if self.current_time_step >= len(self.coverage_indicator):
            return torch.zeros(self._calculate_observation_shape(), device=self.device)

        coverage = self.coverage_indicator[self.current_time_step].flatten().float().to(self.device)
        previous_access_strategy = self.access_decision.flatten().float().to(self.device)
        switch_count = self.switch_count.float().to(self.device)
        elevation_angles = self.eval_angle[self.current_time_step].flatten().float().to(self.device)
        altitudes = self.satellite_heights[self.current_time_step].flatten().float().to(
            self.device)  # 确保这里的处理逻辑是正确的，根据你的数据结构可能需要调整

        observation = torch.cat([coverage, previous_access_strategy, switch_count, elevation_angles, altitudes])
        return observation

    def _calculate_observation_shape(self):
        # 计算每个组成部分的尺寸
        coverage_indicator_numel = self.coverage_indicator[0].numel()
        access_decision_numel = self.access_decision.numel()
        switch_count_numel = self.switch_count.numel()
        eval_angle_numel = self.eval_angle[0].numel()
        satellite_heights_numel = self.satellite_heights[0].numel()
        # 计算总的观测空间尺寸
        total_numel = (coverage_indicator_numel +
                       access_decision_numel +
                       switch_count_numel +
                       eval_angle_numel +
                       satellite_heights_numel)
        shape = torch.Size([total_numel])

        # 打印总的观测空间尺寸
        # print(f"Total observation shape: {shape}")
        return shape

    def calculate_reward(self, action_matrix: torch.Tensor) -> float:
        reward = 0
        #print(self.channel_capacity)
        # 假设 self.channel_capacity 已经不再使用时间维度
        # 对应元素相乘
        result = torch.mul(action_matrix, self.channel_capacity)
        #print(result)
        # 找到张量中非零元素的索引
        nonzero_indices = torch.nonzero(result)

        # 提取出非零元素
        nonzero_elements = result[nonzero_indices[:, 0], nonzero_indices[:, 1]]

        # 对非零元素进行相加
        capacity = torch.sum(nonzero_elements)
        reward += self.w2 * capacity
        # 减少奖励，基于用户切换次数
        # 注意：这里假设 self.switch_count 已经被更新以反映最新的切换情况
        reward -= self.w1 * sum(self.switch_count)

        #print(f"Calculated reward: {reward}")
        return reward

    def calculate_distance_matrix(self) -> torch.Tensor:
        # 获取所有时间段的卫星高度和仰角
        sat_heights = self.satellite_heights  # 假设形状: [61, 301]
        eval_angles = self.eval_angle  # 假设形状: [61, 10, 301]

        # 通过调整形状来启用广播
        sat_heights = sat_heights.unsqueeze(1)  # 形状变为: [61, 1, 301]
        # 注意：这里不再对 eval_angles 进行形状调整，因为它已经是预期形状

        # 计算距离矩阵
        distance = self.radius_earth * (self.radius_earth + sat_heights) / torch.sqrt(
            (self.radius_earth + sat_heights) ** 2 - self.radius_earth ** 2 * torch.cos(torch.deg2rad(eval_angles)) ** 2
        )

        return distance

    def calculate_DL_pathloss_matrix(self, distance_matrix: torch.Tensor) -> torch.Tensor:
        # 计算路径损耗矩阵
        pathloss = 20 * torch.log10(distance_matrix) + 20 * torch.log10(self.communication_frequency) - 147.55

        # print(f"Pathloss matrix shape: {pathloss.shape}")
        return pathloss  # Shape: [NUM_TIME_SLOTS, NUM_SATELLITES, NUM_GROUND_USER]

    #CNR的计算需要根据决策变量来决定，所以应该只记录当前slot下的CNR情况
    def calculate_CNR_matrix(self, time_slot: int, action_matrix: torch.Tensor,
                             distance_matrix: torch.Tensor) -> torch.Tensor:
        # 计算路径损耗矩阵，其形状为 [NUM_TIME_SLOTS, NUM_SATELLITES, NUM_GROUND_USER]
        loss = self.calculate_DL_pathloss_matrix(distance_matrix)

        # 计算接收功率（单位：瓦特），假设 self.EIRP_watts 和 self.receive_benefit_ground 是标量
        received_power_watts = self.EIRP_watts * 10 ** (self.receive_benefit_ground / 10) / (10 ** (loss / 10))
        # print(f"received power watts:",{received_power_watts})

        # 计算 CNR（线性值），假设 self.noise_power 是标量
        CNR_linear = received_power_watts / self.noise_power
        return CNR_linear

    def calculate_interference_matrix(self, time_slot: int, action_matrix: torch.Tensor) -> torch.Tensor:
        interference_matrix = torch.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER), dtype=torch.float32,
                                          device=self.device)
        for user_index in range(self.NUM_GROUND_USER):
            for satellite_index in range(self.NUM_SATELLITES):
                if action_matrix[satellite_index, user_index] == 1:
                    interference_matrix[satellite_index, user_index] = self.calculate_interference(time_slot,
                                                                                                   user_index,
                                                                                                   satellite_index)
        interference_matrix = interference_matrix.transpose(0, 1)
        # print(f"[calculate_interference_matrix] Interference matrix shape: {interference_matrix.shape}")
        return interference_matrix

    def calculate_interference(self, time_slot: int, user_index: int, accessed_satellite_index: int) -> float:
        # 先计算整个时间序列的距离矩阵
        distance_matrix = self.calculate_distance_matrix()
        # 只取当前时间槽的距离矩阵
        current_distance_matrix = distance_matrix[time_slot]

        # 计算路径损耗矩阵，传递正确的距离矩阵
        loss = self.calculate_DL_pathloss_matrix(current_distance_matrix)
        total_interference_power_watts = 0

        for satellite_index in range(self.NUM_SATELLITES):
            if satellite_index != accessed_satellite_index and (
                    self.coverage_indicator[time_slot, user_index, satellite_index, 0] == 1 or
                    self.coverage_indicator[time_slot, user_index, satellite_index, 1] == 1):
                if satellite_index >= loss.shape[0] or user_index >= loss.shape[1]:
                    # print(f"[calculate_interference] Processing satellite_index={accessed_satellite_index}, user_index={user_index}")
                    continue
                loss_value = loss[satellite_index, user_index]
                EIRP_watts = 10 ** ((self.EIRP - 30) / 10)
                interference_power_watts = EIRP_watts * (10 ** (self.receive_benefit_ground / 10)) / (
                        10 ** (loss_value / 10))
                total_interference_power_watts += interference_power_watts

        total_interference_dBm = 10 * torch.log10(torch.tensor(total_interference_power_watts).clone().detach()) + 30
        return total_interference_dBm.item()

    def close(self):
        pass

    def terminate(self) -> Tuple[torch.Tensor, float, bool, dict]:
        observation = torch.zeros(self._calculate_observation_shape(), device=self.device)
        reward = 0.0
        information = {
            'current_time_step': self.current_time_step,
            'switch_count': self.switch_count.clone()
        }
        print("Terminating environment")
        return observation, reward, True, information

    def update_switch_count(self, action_matrix: torch.Tensor):
        if self.current_time_step > 0:
            # 计算当前决策和前一个决策不同的位置
            switch_matrix = (action_matrix != self.previous_access_strategy_space).int()
            # 增加日志记录
            # print(f"Switch matrix:\n{switch_matrix}")
            # 更新切换次数
            if self.current_time_step > 1:  # 假设第一个时隙的切换不计入
                self.switch_count += switch_matrix.sum(dim=0)
            # print(f"Updated switch count: {self.switch_count}")

        # 在每次调用结束时，将当前的接入决策保存为下一次的前一个接入决策
        self.previous_access_strategy_space = action_matrix.clone()

    def update_rates_and_capacity(self, action_matrix: torch.Tensor):
        distance_matrix = self.calculate_distance_matrix()[self.current_time_step]
        # 计算 CNR 矩阵，假设其形状为 [NUM_SATELLITES, NUM_GROUND_USERS]
        CNR = self.calculate_CNR_matrix(self.current_time_step, action_matrix, distance_matrix)
        # print(f"CNR matrix shape: {CNR.shape}, values: {CNR}")

        # 计算 INR 矩阵，假设其形状为 [NUM_SATELLITES, NUM_GROUND_USERS]
        INR = self.calculate_interference_matrix(self.current_time_step, action_matrix)
        # 确保 CNR 和 INR 的形状一致
        assert CNR.shape == INR.shape, f"CNR shape {CNR.shape} does not match INR shape {INR.shape}"

        # 直接更新信道容量，不考虑时间维度
        self.channel_capacity = self.total_bandwidth * torch.log2(1.0 + CNR / (INR + 1.0))
        #print(self.channel_capacity)
        # 确保 channel_capacity 形状正确
        if self.channel_capacity.shape != (self.NUM_SATELLITES, self.NUM_GROUND_USER):
            self.channel_capacity = self.channel_capacity.transpose(0, 1)
