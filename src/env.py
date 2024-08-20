import pandas as pd
import torch
import os
from typing import Optional, Tuple


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
        self.w1 = 1  # 切换次数的权重
        self.w2 = 1  # 用户传输速率的权重
        self.r_thr = -5  # 最低的CINR阈值，单位：dB

        # 定义动作空间和观察空间
        self.action_space = torch.zeros(self.NUM_SATELLITES * self.NUM_GROUND_USER, dtype=torch.int)
        self.coverage_space = torch.zeros((self.NUM_TIME_SLOTS, self.NUM_SATELLITES, self.NUM_GROUND_USER, 2), dtype=torch.int)
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
        print(f"Moved to device: {device}")

    def initialize_angle(self):
        # 从CSV文件读取地面用户的仰角数据
        df = pd.read_csv('ev_data.csv')
        eval_angle = torch.zeros((self.NUM_TIME_SLOTS, self.NUM_GROUND_USER, self.NUM_SATELLITES), dtype=torch.float32)

        # 填充 eval_angle 数组
        for time_slot in range(self.NUM_TIME_SLOTS):
            for i in range(self.NUM_SATELLITES):
                for j in range(self.NUM_GROUND_USER):
                    eval_angle[time_slot, j, i] = df.iloc[i * self.NUM_GROUND_USER + j, time_slot]

        print(f"Initialized eval_angle with shape: {eval_angle.shape}")  # [TIME_SLOTS,NUM_GROUND_USER,NUM_SATELLITES]
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

        print(f"alt DataFrame shape: {df.shape}")

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

        print(f"Initialized sat_heights with shape: {sat_heights.shape}")  # [NUM_TIME_SLOTS, NUM_SATELLITES]
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
        print(f"cov DataFrame shape: {df.shape}")

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

        print(f"Initialized coverage with shape: {coverage.shape}")
        return coverage

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        # 确保 action 是张量
        action = self.ensure_tensor(action)
        print(f"Action shape: {action.shape}")

        # 确保 action 的形状正确
        action_matrix = action.view((self.NUM_SATELLITES, self.NUM_GROUND_USER)).to(self.device)
        print(f"Action matrix shape: {action_matrix.shape}")

        # 检查是否已经达到最大时间步数
        if self.current_time_step >= self.NUM_TIME_SLOTS:
            return self.terminate()

        # 更新接入决策变量
        self.access_decision = action_matrix
        # 检查并更新切换次数
        if self.current_time_step > 0:
            self.update_switch_count(action_matrix)

        # 计算用户传输速率和信道容量
        self.update_rates_and_capacity(action_matrix)

        # 计算奖励
        reward = self.calculate_reward(action_matrix)
        print(f"Reward: {reward}")

        # 更新时间步
        self.current_time_step += 1

        # 检查是否结束
        is_done = self.current_time_step >= self.NUM_TIME_SLOTS

        # 获取观察
        observation = self.get_observation() if not is_done else torch.zeros(self.get_observation_shape(),
                                                                             device=self.device)

        # 可选的额外信息
        information = {
            'current_time_step': self.current_time_step,
            'switch_count': self.switch_count.clone()
        }

        return observation, reward, is_done, information

    def initialize_coverage_indicator(self):
        for time_slot in range(self.NUM_TIME_SLOTS):
            for user_index in range(self.NUM_GROUND_USER):
                for satellite_index in range(self.NUM_SATELLITES):
                    if self.eval_angle[time_slot, user_index, satellite_index] > self.angle_threshold:
                        self.coverage_indicator[time_slot, user_index, satellite_index] = 1
                    else:
                        self.coverage_indicator[time_slot, user_index, satellite_index] = 0
        print(f"Initialized coverage indicator with shape: {self.coverage_indicator.shape}")

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[torch.Tensor, dict]:
        # 重置当前时间步
        self.current_time_step = 0

        # 重置覆盖指示变量和接入决策变量
        self.coverage_indicator = torch.zeros((self.NUM_TIME_SLOTS, self.NUM_GROUND_USER, self.NUM_SATELLITES),
                                              dtype=torch.int, device=self.device)
        self.access_decision = torch.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER), dtype=torch.int,
                                           device=self.device)

        # 初始化覆盖指示变量
        self.initialize_coverage_indicator()

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
        self.user_demand_rate.uniform_(1e9, 10e9)

        # 获取初始观察
        observation = self.get_observation()
        print(f"Reset observation shape: {observation.shape}")

        return observation, {'current_time_step': self.current_time_step}

    def get_observation(self) -> torch.Tensor:
        if self.current_time_step >= len(self.coverage_indicator):
            return torch.zeros(self._calculate_observation_shape(), device=self.device)

        coverage = self.coverage_indicator[self.current_time_step].flatten().float()
        previous_access_strategy = self.access_decision[:, :,
                                   self.current_time_step - 1].flatten().float() if self.current_time_step > 0 else torch.zeros(
            (self.NUM_SATELLITES, self.NUM_GROUND_USER), device=self.device).flatten().float()
        switch_count = self.switch_count.float()
        elevation_angles = self.eval_angle[self.current_time_step].flatten().float()
        altitudes = self.satellite_heights[self.current_time_step].float()

        observation = torch.cat([coverage, previous_access_strategy, switch_count, elevation_angles, altitudes])
        print(f"Observation concatenated shape: {observation.shape}")
        return observation

    def _calculate_observation_shape(self):
        shape = torch.cat([
            self.coverage_indicator[0].flatten().float(),
            torch.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER), device=self.device).flatten().float(),
            self.switch_count.float(),
            self.eval_angle[0].flatten().float(),
            self.satellite_heights[0].float()
        ]).shape
        print(f"Observation shape: {shape}")
        return shape

    def calculate_reward(self, action_matrix: torch.Tensor) -> float:
        reward = 0
        for satellite_index in range(self.NUM_SATELLITES):
            for user_index in range(self.NUM_GROUND_USER):
                if action_matrix[satellite_index, user_index] == 1:
                    rate = self.user_rate[satellite_index, user_index, self.current_time_step]
                    if rate >= self.r_thr:
                        reward += self.w2 * rate
                    else:
                        reward -= self.w1 * self.switch_count[user_index]
        print(f"Calculated reward: {reward}")
        return reward

    def calculate_distance_matrix(self) -> torch.Tensor:
        # 获取所有时间段的卫星高度和仰角
        sat_heights = self.satellite_heights  # Shape: [NUM_TIME_SLOTS, NUM_SATELLITES]
        eval_angles = self.eval_angle  # Shape: [NUM_TIME_SLOTS, NUM_GROUND_USER, NUM_SATELLITES]

        # Reshape to enable broadcasting
        sat_heights = sat_heights.unsqueeze(1).unsqueeze(3)  # Shape: [NUM_TIME_SLOTS, 1, NUM_SATELLITES, 1]
        eval_angles = eval_angles.unsqueeze(2)  # Shape: [NUM_TIME_SLOTS, NUM_GROUND_USER, 1, NUM_SATELLITES]

        # 计算距离矩阵
        distance = self.radius_earth * (self.radius_earth + sat_heights) / torch.sqrt(
            (self.radius_earth + sat_heights) ** 2 - self.radius_earth ** 2 * torch.cos(torch.deg2rad(eval_angles)) ** 2
        )

        # 调整形状为 [NUM_TIME_SLOTS, NUM_SATELLITES, NUM_GROUND_USER]
        distance = distance.squeeze(3).permute(0, 2, 1)  # Shape: [NUM_TIME_SLOTS, NUM_SATELLITES, NUM_GROUND_USER]

        print(f"Distance matrix shape: {distance.shape}")
        return distance  # Shape: [NUM_TIME_SLOTS, NUM_SATELLITES, NUM_GROUND_USER]

    def calculate_DL_pathloss_matrix(self, distance_matrix: torch.Tensor) -> torch.Tensor:
        # 计算路径损耗矩阵
        pathloss = 20 * torch.log10(distance_matrix) + 20 * torch.log10(self.communication_frequency) - 147.55

        print(f"Pathloss matrix shape: {pathloss.shape}")
        return pathloss  # Shape: [NUM_TIME_SLOTS, NUM_SATELLITES, NUM_GROUND_USER]

    #CNR的计算需要根据决策变量来决定，所以应该只记录当前slot下的CNR情况
    def calculate_CNR_matrix(self, time_slot: int, action_matrix: torch.Tensor, distance_matrix: torch.Tensor) -> torch.Tensor:
        # 计算路径损耗矩阵，其形状为 [NUM_TIME_SLOTS, NUM_SATELLITES, NUM_GROUND_USER]
        loss = self.calculate_DL_pathloss_matrix(distance_matrix)

        # 计算接收功率（单位：瓦特），假设 self.EIRP_watts 和 self.receive_benefit_ground 是标量
        received_power_watts = self.EIRP_watts * 10 ** (self.receive_benefit_ground / 10) / (10 ** (loss / 10))

        # 计算 CNR（线性值），假设 self.noise_power 是标量
        CNR_linear = received_power_watts / self.noise_power

        # 返回 CNR 的对数值（单位：dB），保持矩阵形状
        CNR = 10 * torch.log10(CNR_linear)
        print(f"CNR matrix shape: {CNR.shape}")
        return CNR

    def calculate_interference_matrix(self, time_slot: int, action_matrix: torch.Tensor) -> torch.Tensor:
        interference_matrix = torch.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER), dtype=torch.float32, device=self.device)
        for user_index in range(self.NUM_GROUND_USER):
            for satellite_index in range(self.NUM_SATELLITES):
                if action_matrix[satellite_index, user_index] == 1:
                    interference_matrix[satellite_index, user_index] = self.calculate_interference(time_slot, user_index, satellite_index)
        print(f"Interference matrix shape: {interference_matrix.shape}")
        return interference_matrix

    def calculate_interference(self, time_slot: int, user_index: int, accessed_satellite_index: int) -> float:
        total_interference_power_watts = 0
        for satellite_index in range(self.NUM_SATELLITES):
            if satellite_index != accessed_satellite_index and self.coverage_indicator[
                time_slot, user_index, satellite_index] == 1:
                loss = self.calculate_DL_pathloss_matrix(torch.tensor(time_slot))
                if satellite_index >= loss.shape[0] or user_index >= loss.shape[1]:
                    print(
                        f"Index out of bounds: satellite_index={satellite_index}, user_index={user_index}, loss.shape={loss.shape}")
                    continue
                loss_value = loss[satellite_index, user_index]
                EIRP_watts = 10 ** ((self.EIRP - 30) / 10)
                interference_power_watts = EIRP_watts * (10 ** (self.receive_benefit_ground / 10)) / (
                            10 ** (loss_value / 10))
                total_interference_power_watts += interference_power_watts

        total_interference_dBm = 10 * torch.log10(torch.tensor(total_interference_power_watts).clone().detach()) + 30
        print(f"Calculated interference: {total_interference_dBm.item()} dBm")
        return total_interference_dBm.item()

    def update_coverage_indicator(self, current_time_slot: int):
        for user_index in range(self.NUM_GROUND_USER):
            for satellite_index in range(self.NUM_SATELLITES):
                if self.eval_angle[current_time_slot, user_index, satellite_index] > self.angle_threshold:
                    self.coverage_indicator[current_time_slot, user_index, satellite_index] = 1
                else:
                    self.coverage_indicator[current_time_slot, user_index, satellite_index] = 0
        print(f"Updated coverage indicator for time slot {current_time_slot}")

    def calculate_actual_rate_matrix(self, time_slot: int, action_matrix: torch.Tensor) -> torch.Tensor:
        capacity = self.channel_capacity[:, :, time_slot]
        demand = self.user_demand_rate[:, time_slot]
        actual_rate = torch.min(capacity, demand)
        print(f"Actual rate matrix shape: {actual_rate.shape}")
        return actual_rate

    def render(self, mode='human'):
        print(f"Current time step: {self.current_time_step}")

    def close(self):
        pass

    def ensure_tensor(self, data) -> torch.Tensor:
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.int).to(self.device)
        print(f"Ensured tensor shape: {data.shape}")
        return data

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
        previous_action_matrix = self.access_decision[:, :, self.current_time_step - 1]
        switch_matrix = (action_matrix != previous_action_matrix).int()
        self.switch_count += switch_matrix.sum(dim=0)
        print(f"Updated switch count: {self.switch_count}")

    def update_rates_and_capacity(self, action_matrix: torch.Tensor):
        # 计算 CNR 矩阵，假设其形状为 [NUM_SATELLITES, NUM_GROUND_USERS]
        CNR = self.calculate_CNR_matrix(self.current_time_step, action_matrix)
        print(f"CNR matrix shape: {CNR.shape}, values: {CNR}")

        # 计算 INR 矩阵，假设其形状为 [NUM_SATELLITES, NUM_GROUND_USERS]
        INR = self.calculate_interference_matrix(self.current_time_step, action_matrix)
        print(f"INR matrix shape: {INR.shape}, values: {INR}")

        # 确保 CNR 和 INR 的形状一致
        assert CNR.shape == INR.shape, f"CNR shape {CNR.shape} does not match INR shape {INR.shape}"

        # 更新信道容量，假设 self.channel_capacity 的形状为 [NUM_SATELLITES, NUM_GROUND_USERS, TIME_STEPS]
        self.channel_capacity[:, :, self.current_time_step] = self.total_bandwidth * torch.log2(1.0 + CNR / (INR + 1.0))
        print(
            f"Updated channel capacity for time slot {self.current_time_step}: {self.channel_capacity[:, :, self.current_time_step]}")

        # 更新用户速率，假设 self.user_rate 的形状为 [NUM_SATELLITES, NUM_GROUND_USERS, TIME_STEPS]
        self.user_rate[:, :, self.current_time_step] = self.calculate_actual_rate_matrix(self.current_time_step,
                                                                                         action_matrix)
        print(
            f"Updated user rate for time slot {self.current_time_step}: {self.user_rate[:, :, self.current_time_step]}")
