import numpy as np
import pandas as pd
import torch
from typing import Optional, Tuple


class Env:
    def __init__(self):
        super(Env, self).__init__()
        # 定义卫星和用户的数量
        self.NUM_SATELLITES = 300  # 卫星数量
        self.NUM_GROUND_USER = 10  # 地面用户数量
        self.TOTAL_TIME = 3000  # 总模拟时间，单位：秒
        self.NUM_TIME_SLOTS = 60  # 时间段的划分数量
        self.TIME_SLOT_DURATION = self.TOTAL_TIME // self.NUM_TIME_SLOTS  # 每个时间段的持续时间
        self.communication_frequency = 18.5e9  # 通信频率为18.5GHz
        self.total_bandwidth = 250e6  # 总带宽为250MHz
        self.noise_temperature = 213.15  # 系统的噪声温度为213.15开尔文
        self.Polarization_isolation_factor = 12  # 单位dB
        self.receive_benefit_ground = 15.4  # 单位dB
        self.EIRP = 73.1  # 单位:dBm
        self.k = 1.380649e-23  # 单位:J/K
        self.radius_earth = 6731e3  # 单位:m
        self.EIRP_watts = 10 ** ((self.EIRP - 30) / 10)  # 将 EIRP 从 dBm 转换为瓦特
        self.noise_power = self.k * self.noise_temperature * self.total_bandwidth  # 噪声功率计算

        # 定义动作空间和观察空间
        self.action_space = torch.zeros(self.NUM_SATELLITES * self.NUM_GROUND_USER,
                                        dtype=torch.int)  # 动作空间的形状：300*10，每个卫星选择对哪个地面站进行接入

        # 调整观察空间以匹配实际的观察形状
        self.coverage_space = torch.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER), dtype=torch.int)
        self.previous_access_strategy_space = torch.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER), dtype=torch.int)
        self.switch_count_space = torch.zeros(self.NUM_GROUND_USER, dtype=torch.int)  # 假设最大切换次数为100
        self.elevation_angle_space = torch.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER), dtype=torch.float32)
        self.altitude_space = torch.zeros(self.NUM_SATELLITES, dtype=torch.float32)
        self.observation_space = {
            "coverage": self.coverage_space,
            "previous_access_strategy": self.previous_access_strategy_space,
            "switch_count": self.switch_count_space,
            "elevation_angles": self.elevation_angle_space,
            "altitudes": self.altitude_space
        }

        # 初始化卫星和用户的位置
        self.satellite_heights = self.initialize_altitude()  # 卫星高度这里的单位时km
        self.eval_angle = self.initialize_angle()  # 俯仰角的单位是度
        self.angle_threshold = 15  # 单位：度

        # 初始化覆盖指示变量和接入决策变量
        self.coverage_indicator = torch.zeros((self.NUM_TIME_SLOTS, self.NUM_GROUND_USER, self.NUM_SATELLITES),
                                              dtype=torch.int)
        self.access_decision = torch.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER, self.NUM_TIME_SLOTS),
                                           dtype=torch.int)
        self.current_time_step = 0
        self.w1 = 1  # 切换次数的权重
        self.w2 = 1  # 用户传输速率的权重
        self.r_thr = -5  # 最低的CINR阈值，单位：dB
        self.switch_count = torch.zeros(self.NUM_GROUND_USER, dtype=torch.int)  # 每个用户的切换次数
        # 初始化用户传输速率矩阵
        self.user_rate = torch.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER, self.NUM_TIME_SLOTS),
                                     dtype=torch.float32)

        # 初始化信道容量矩阵
        self.channel_capacity = torch.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER, self.NUM_TIME_SLOTS),
                                            dtype=torch.float32)

        # 初始化用户需求速率
        self.user_demand_rate = torch.tensor(np.random.uniform(1e6, 10e6, (self.NUM_GROUND_USER, self.NUM_TIME_SLOTS)),
                                             dtype=torch.float32)  # 随机初始化需求速率

        # 使用 GPU 加速
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        print("Environment initialized")

    def to(self, device):
        self.coverage_indicator = self.coverage_indicator.to(device)
        self.access_decision = self.access_decision.to(device)
        self.switch_count = self.switch_count.to(device)
        self.user_rate = self.user_rate.to(device)
        self.channel_capacity = self.channel_capacity.to(device)
        self.user_demand_rate = self.user_demand_rate.to(device)
        self.satellite_heights = self.satellite_heights.to(device)
        self.eval_angle = self.eval_angle.to(device)

    def initialize_angle(self):
        # 从CSV文件读取地面用户的仰角数据
        df = pd.read_csv('ev_data.csv')
        eval_angle = torch.zeros((self.NUM_TIME_SLOTS, self.NUM_GROUND_USER, self.NUM_SATELLITES), dtype=torch.float32)

        # 填充 eval_angle 数组
        for time_slot in range(self.NUM_TIME_SLOTS):
            for i in range(self.NUM_SATELLITES):
                for j in range(self.NUM_GROUND_USER):
                    eval_angle[time_slot, j, i] = df.iloc[1 + i * self.NUM_GROUND_USER + j, time_slot]

        return eval_angle

    def initialize_altitude(self):
        # 从CSV文件读取卫星的高度数据
        df = pd.read_csv('alt_data.csv')
        sat_heights = torch.zeros((self.NUM_TIME_SLOTS, self.NUM_SATELLITES), dtype=torch.float32)

        # 填充 sat_heights 数组
        for time_slot in range(self.NUM_TIME_SLOTS):
            for i in range(self.NUM_SATELLITES):
                sat_heights[time_slot, i] = df.iloc[1 + i, time_slot]

        return sat_heights

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        # 检查是否已经达到最大时间步数
        if self.current_time_step >= self.NUM_TIME_SLOTS:
            is_done = True
            observation = torch.zeros(self.get_observation_shape(), device=self.device)
            reward = 0.0
            information = {
                'current_time_step': self.current_time_step,
                'switch_count': self.switch_count.clone()
            }
            return observation, reward, is_done, information

        # 更新覆盖指示变量
        self.update_coverage_indicator(self.current_time_step)

        # 将一维动作数组转换为二维形式，用于表示每个用户选择的卫星
        action_matrix = action.view((self.NUM_SATELLITES, self.NUM_GROUND_USER))

        # 检查并更新切换次数
        if self.current_time_step > 0:
            previous_action_matrix = self.access_decision[:, :, self.current_time_step - 1]
            switch_matrix = (action_matrix != previous_action_matrix).int()
            self.switch_count += switch_matrix.sum(dim=0)

        # 根据action更新接入决策
        self.access_decision[:, :, self.current_time_step] = action_matrix

        # 计算用户传输速率和信道容量
        CNR = self.calculate_CNR_matrix(self.current_time_step, action_matrix)
        INR = self.calculate_interference_matrix(self.current_time_step, action_matrix)
        self.channel_capacity[:, :, self.current_time_step] = self.total_bandwidth * torch.log2(1.0 + CNR / (INR + 1.0))
        self.user_rate[:, :, self.current_time_step] = self.calculate_actual_rate_matrix(self.current_time_step,
                                                                                         action_matrix)

        # 计算奖励
        reward = self.calculate_reward(action_matrix)

        # 更新时间步
        self.current_time_step += 1

        # 检查是否结束
        is_done = self.current_time_step >= self.NUM_TIME_SLOTS

        if is_done:
            observation = torch.zeros(self.get_observation_shape(), device=self.device)
        else:
            # 获取当前环境的观察
            observation = self.get_observation()

        # 可选的额外信息
        information = {
            'current_time_step': self.current_time_step,
            'switch_count': self.switch_count.clone()
        }

        # 设置 terminated 和 truncated
        is_terminated = is_done

        return observation, reward, is_terminated, information

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[torch.Tensor, dict]:
        # 重置当前时间步
        self.current_time_step = 0

        # 重置覆盖指示变量和接入决策变量
        self.coverage_indicator = torch.zeros((self.NUM_TIME_SLOTS, self.NUM_GROUND_USER, self.NUM_SATELLITES),
                                              dtype=torch.int, device=self.device)
        self.access_decision = torch.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER, self.NUM_TIME_SLOTS),
                                           dtype=torch.int, device=self.device)

        # 重置切换次数
        self.switch_count = torch.zeros(self.NUM_GROUND_USER, dtype=torch.int, device=self.device)

        # 重置用户传输速率
        self.user_rate = torch.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER, self.NUM_TIME_SLOTS),
                                     dtype=torch.float32, device=self.device)

        # 重置信道容量矩阵
        self.channel_capacity = torch.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER, self.NUM_TIME_SLOTS),
                                            dtype=torch.float32, device=self.device)

        # 重置用户需求速率
        self.user_demand_rate = torch.tensor(np.random.uniform(1e6, 10e6, (self.NUM_GROUND_USER, self.NUM_TIME_SLOTS)),
                                             dtype=torch.float32, device=self.device)  # 随机初始化需求速率

        # 获取初始观察
        observation = self.get_observation()

        return observation, {'current_time_step': self.current_time_step}

    def get_observation(self) -> torch.Tensor:
        if self.current_time_step >= len(self.coverage_indicator):
            # 返回一个默认的观察值，而不是抛出异常
            return torch.zeros(self.get_observation_shape(), device=self.device)

        coverage = self.coverage_indicator[self.current_time_step].flatten().float()
        previous_access_strategy = self.access_decision[:, :,
                                   self.current_time_step - 1].flatten().float() if self.current_time_step > 0 else torch.zeros(
            (self.NUM_SATELLITES, self.NUM_GROUND_USER), device=self.device).flatten().float()
        switch_count = self.switch_count.float()
        elevation_angles = self.eval_angle[self.current_time_step].flatten().float()
        altitudes = self.satellite_heights[self.current_time_step].float()

        observation = torch.cat([coverage, previous_access_strategy, switch_count, elevation_angles, altitudes])
        return observation

    def get_observation_shape(self):
        # 获取展平后的观察空间的形状
        return torch.cat([
            self.coverage_indicator[0].flatten().float(),
            torch.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER), device=self.device).flatten().float(),
            self.switch_count.float(),
            self.eval_angle[0].flatten().float(),
            self.satellite_heights[0].float()
        ]).shape

    def calculate_reward(self, action_matrix: torch.Tensor) -> float:
        # 计算奖励
        reward = 0
        for satellite_index in range(self.NUM_SATELLITES):
            for user_index in range(self.NUM_GROUND_USER):
                if action_matrix[satellite_index, user_index] == 1:
                    # 计算用户传输速率
                    rate = self.user_rate[satellite_index, user_index, self.current_time_step]
                    # 检查是否满足最低CNR阈值
                    if rate >= self.r_thr:
                        reward += self.w2 * rate
                    else:
                        reward -= self.w1 * self.switch_count[user_index]

        return reward

    def calculate_distance_matrix(self, time_slot: int) -> torch.Tensor:
        sat_height = self.satellite_heights[time_slot]
        eval_angle = self.eval_angle[time_slot]
        return self.radius_earth * (self.radius_earth + sat_height) / torch.sqrt(
            (self.radius_earth + sat_height) ** 2 - self.radius_earth ** 2 * torch.cos(torch.deg2rad(eval_angle)) ** 2)

    def calculate_DL_pathloss_matrix(self, time_slot: int) -> torch.Tensor:
        distance = self.calculate_distance_matrix(time_slot)
        return 20 * torch.log10(distance) + 20 * torch.log10(torch.tensor(self.communication_frequency)) - 147.55

    def calculate_CNR_matrix(self, time_slot: int, action_matrix: torch.Tensor) -> torch.Tensor:
        loss = self.calculate_DL_pathloss_matrix(time_slot)
        received_power_watts = self.EIRP_watts * 10 ** (self.receive_benefit_ground / 10) / (10 ** (loss / 10))
        CNR_linear = received_power_watts / self.noise_power
        return 10 * torch.log10(CNR_linear)

    def calculate_interference(self, time_slot: int, user_index: int, accessed_satellite_index: int) -> float:
        # 初始化总干扰功率为0
        total_interference_power_watts = 0

        # 遍历所有卫星，计算每个卫星对用户的干扰
        for satellite_index in range(self.NUM_SATELLITES):
            # 检查卫星是否不是当前接入的卫星并且在用户的覆盖范围内
            if satellite_index != accessed_satellite_index and self.coverage_indicator[
                time_slot, user_index, satellite_index] == 1:
                # 计算从该卫星到用户的下行路径损耗
                loss = self.calculate_DL_pathloss(time_slot, user_index, satellite_index)
                # 将 EIRP 从 dBm 转换为瓦特，以便进行线性计算
                EIRP_watts = 10 ** ((self.EIRP - 30) / 10)
                # 计算该卫星产生的干扰功率
                interference_power_watts = EIRP_watts * (10 ** (self.receive_benefit_ground / 10)) / (10 ** (loss / 10))
                # 累加总干扰功率
                total_interference_power_watts += interference_power_watts

        # 将总干扰功率从瓦特转换为 dBm，以便与其他以 dBm 为单位的参数进行比较
        # 注意：转换回 dBm 需要加上 30 dBm (因为 1W = 30 dBm)
        total_interference_dBm = 10 * torch.log10(torch.tensor(total_interference_power_watts)) + 30

        # 返回总干扰功率的 dBm 值
        return total_interference_dBm.item()

    def update_coverage_indicator(self, current_time_slot: int):
        for user_index in range(self.NUM_GROUND_USER):
            for satellite_index in range(self.NUM_SATELLITES):
                # 检查仰角是否大于限定值
                if self.eval_angle[current_time_slot, user_index, satellite_index] > self.angle_threshold:
                    self.coverage_indicator[current_time_slot, user_index, satellite_index] = 1
                else:
                    self.coverage_indicator[current_time_slot, user_index, satellite_index] = 0

    def calculate_actual_rate_matrix(self, time_slot: int, action_matrix: torch.Tensor) -> torch.Tensor:
        capacity = self.channel_capacity[:, :, time_slot]
        demand = self.user_demand_rate[:, time_slot]
        return torch.min(capacity, demand)

    def render(self, mode='human'):
        """
        渲染环境。
        """
        # 示例：打印当前时间步
        print(f"Current time step: {self.current_time_step}")

    def close(self):
        """
        关闭环境。
        """
        pass
