import numpy as np
import gymnasium as gym
import pandas as pd
from gymnasium import spaces
from gymnasium.spaces import Box, MultiBinary
from typing import Optional, Tuple


class Env(gym.Env):
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

        # 定义动作空间和观察空间
        self.action_space = MultiBinary(self.NUM_SATELLITES * self.NUM_GROUND_USER)  # 动作空间的形状：300*10，每个卫星选择对哪个地面站进行接入

        # 调整观察空间以匹配实际的观察形状
        # observation_shape = (self.NUM_SATELLITES * self.NUM_GROUND_USER + self.NUM_GROUND_USER,)  # 观察空间应该至少包括：卫星对用户的覆盖，
        self.coverage_space = spaces.MultiBinary((self.NUM_SATELLITES, self.NUM_GROUND_USER))
        self.previous_access_strategy_space = spaces.MultiBinary((self.NUM_SATELLITES, self.NUM_GROUND_USER))
        self.switch_count_space = spaces.Discrete(100)  # 假设最大切换次数为100
        self.elevation_angle_space = spaces.Box(low=-90, high=90, shape=(self.NUM_SATELLITES, self.NUM_GROUND_USER),
                                                dtype=np.float32)
        self.altitude_space = spaces.Box(low=0, high=10000, shape=(self.NUM_SATELLITES,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "coverage": self.coverage_space,
            "previous_access_strategy": self.previous_access_strategy_space,
            "switch_count": spaces.MultiDiscrete([self.switch_count_space.n] * self.NUM_GROUND_USER),
            "elevation_angles": self.elevation_angle_space,
            "altitudes": self.altitude_space
        })

        # 初始化卫星和用户的位置
        self.satellite_heights = self.initialize_altitude()  # 卫星高度这里的单位时km
        self.eval_angle = self.initialize_angle()  # 俯仰角的单位是度
        self.angle_threshold = 15  # 单位：度

        # 初始化覆盖指示变量和接入决策变量
        self.coverage_indicator = np.zeros((self.NUM_TIME_SLOTS, self.NUM_GROUND_USER, self.NUM_SATELLITES))
        self.access_decision = np.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER, self.NUM_TIME_SLOTS))
        self.current_time_step = 0
        self.w1 = 1  # 切换次数的权重
        self.w2 = 1  # 用户传输速率的权重
        self.r_thr = -5  # 最低的CINR阈值，单位：dB
        self.switch_count = np.zeros(self.NUM_GROUND_USER)  # 每个用户的切换次数
        # 初始化用户传输速率矩阵
        self.user_rate = np.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER, self.NUM_TIME_SLOTS))

        # 初始化信道容量矩阵
        self.channel_capacity = np.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER, self.NUM_TIME_SLOTS))

        # 初始化用户需求速率
        self.user_demand_rate = np.random.uniform(1e6, 10e6, (self.NUM_GROUND_USER, self.NUM_TIME_SLOTS))  # 随机初始化需求速率

        print("Environment initialized")

    def initialize_angle(self):
        # 从CSV文件读取地面用户的仰角数据
        df = pd.read_csv('ev_data.csv')
        eval_angle = np.zeros((self.NUM_TIME_SLOTS, self.NUM_GROUND_USER, self.NUM_SATELLITES))

        # 填充 eval_angle 数组
        for time_slot in range(self.NUM_TIME_SLOTS):
            for i in range(self.NUM_SATELLITES):
                for j in range(self.NUM_GROUND_USER):
                    eval_angle[time_slot, j, i] = df.iloc[1 + i * self.NUM_GROUND_USER + j, time_slot]

        return eval_angle

    def initialize_altitude(self):
        # 从CSV文件读取卫星的高度数据
        df = pd.read_csv('alt_data.csv')
        sat_heights = np.zeros((self.NUM_TIME_SLOTS, self.NUM_SATELLITES))

        # 填充 sat_heights 数组
        for time_slot in range(self.NUM_TIME_SLOTS):
            for i in range(self.NUM_SATELLITES):
                sat_heights[time_slot, i] = df.iloc[1 + i, time_slot]

        return sat_heights

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # 检查是否已经达到最大时间步数
        if self.current_time_step >= self.NUM_TIME_SLOTS:
            is_done = True
            observation = np.zeros(self.observation_space.shape)
            reward = 0.0
            information = {
                'current_time_step': self.current_time_step,
                'switch_count': self.switch_count.copy()
            }
            return observation, reward, is_done, False, information

        # 更新覆盖指示变量
        self.update_coverage_indicator(self.current_time_step)

        # 将一维动作数组转换为二维形式，用于表示每个用户选择的卫星
        action_matrix = action.reshape((self.NUM_SATELLITES, self.NUM_GROUND_USER))

        # 检查并更新切换次数
        for user_index in range(self.NUM_GROUND_USER):
            current_satellite = np.argmax(action_matrix[:, user_index])
            if self.current_time_step > 0:
                previous_satellite = np.argmax(self.access_decision[:, user_index, self.current_time_step - 1])
                if current_satellite != previous_satellite:
                    self.switch_count[user_index] += 1

        # 根据action更新接入决策
        self.access_decision[:, :, self.current_time_step] = action_matrix

        # 计算用户传输速率和信道容量
        for user_index in range(self.NUM_GROUND_USER):
            for satellite_index in range(self.NUM_SATELLITES):
                if action_matrix[satellite_index, user_index] == 1:
                    CNR = self.calculate_CNR(self.current_time_step, user_index, satellite_index)
                    INR = self.calculate_interference(self.current_time_step, user_index, satellite_index)
                    self.channel_capacity[
                        satellite_index, user_index, self.current_time_step] = self.total_bandwidth * np.log2(
                        1 + CNR / (INR + 1))
                    self.user_rate[satellite_index, user_index, self.current_time_step] = self.calculate_actual_rate(
                        self.current_time_step, user_index, satellite_index)

        # 计算奖励
        reward = self.calculate_reward(action_matrix)

        # 更新时间步
        self.current_time_step += 1

        # 检查是否结束
        is_done = self.current_time_step >= self.NUM_TIME_SLOTS

        if is_done:
            observation = np.zeros(self.observation_space.shape)
        else:
            # 获取当前环境的观察
            observation = self.get_observation()

        # 可选的额外信息
        information = {
            'current_time_step': self.current_time_step,
            'switch_count': self.switch_count.copy()
        }

        # 设置 terminated 和 truncated
        is_terminated = is_done
        is_truncated = False

        return observation, reward, is_terminated, is_truncated, information

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        # 调用父类的reset方法以处理随机种子
        super().reset(seed=seed)

        # 重置当前时间步
        self.current_time_step = 0

        # 重置覆盖指示变量和接入决策变量
        self.coverage_indicator = np.zeros((self.NUM_TIME_SLOTS, self.NUM_GROUND_USER, self.NUM_SATELLITES))
        self.access_decision = np.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER, self.NUM_TIME_SLOTS))

        # 重置切换次数
        self.switch_count = np.zeros(self.NUM_GROUND_USER)

        # 重置用户传输速率
        self.user_rate = np.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER, self.NUM_TIME_SLOTS))

        # 重置信道容量矩阵
        self.channel_capacity = np.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER, self.NUM_TIME_SLOTS))

        # 重置用户需求速率
        self.user_demand_rate = np.random.uniform(1e6, 10e6, (self.NUM_GROUND_USER, self.NUM_TIME_SLOTS))  # 随机初始化需求速率

        # 获取初始观察
        observation = self.get_observation()

        return observation, {'current_time_step': self.current_time_step}

    def get_observation(self) -> np.ndarray:
        coverage = self.coverage_indicator[self.current_time_step].flatten()
        previous_access_strategy = self.access_decision[:, :,
                                   self.current_time_step - 1].flatten() if self.current_time_step > 0 else np.zeros(
            (self.NUM_SATELLITES, self.NUM_GROUND_USER)).flatten()
        switch_count = self.switch_count
        elevation_angles = self.eval_angle[self.current_time_step].flatten()
        altitudes = self.satellite_heights[self.current_time_step]

        observation = np.concatenate([coverage, previous_access_strategy, switch_count, elevation_angles, altitudes])
        return observation

    def calculate_reward(self, action_matrix: np.ndarray) -> float:
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

    def calculate_distance(self, time_slot: int, user_index: int, satellite_index: int) -> float:
        # 使用指定时间槽的卫星高度和俯仰角
        sat_height = self.satellite_heights[time_slot, satellite_index]
        eval_angle = self.eval_angle[time_slot, user_index, satellite_index]
        # 计算距离
        result = self.radius_earth * (self.radius_earth + sat_height) / np.sqrt(
            np.pow((self.radius_earth + sat_height), 2) - self.radius_earth ** 2 * np.cos(np.radians(eval_angle)) ** 2)
        return result

    def calculate_DL_pathloss(self, time_slot: int, user_index: int, satellite_index: int) -> float:
        distance = self.calculate_distance(time_slot, user_index, satellite_index)
        result = 20 * np.log10(distance) + 20 * np.log10(self.communication_frequency) - 147.55
        return result

    def calculate_CNR(self, time_slot: int, user_index: int, satellite_index: int) -> float:
        loss = self.calculate_DL_pathloss(time_slot, user_index, satellite_index)
        EIRP_watts = 10 ** ((self.EIRP - 30) / 10)  # 将 EIRP 从 dBm 转换为瓦特
        noise_power = self.k * self.noise_temperature * self.total_bandwidth  # 噪声功率计算
        received_power_watts = EIRP_watts * 10 ** (self.receive_benefit_ground / 10) / (10 ** (loss / 10))  # 接收功率
        CNR_linear = received_power_watts / noise_power  # 线性单位的载噪比
        result = 10 * np.log10(CNR_linear)  # 转换为 dB
        return result

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
        total_interference_dBm = 10 * np.log10(total_interference_power_watts) + 30

        # 返回总干扰功率的 dBm 值
        return total_interference_dBm

    def update_coverage_indicator(self, current_time_slot: int):
        for user_index in range(self.NUM_GROUND_USER):
            for satellite_index in range(self.NUM_SATELLITES):
                # 检查仰角是否大于限定值
                if self.eval_angle[current_time_slot, user_index, satellite_index] > self.angle_threshold:
                    self.coverage_indicator[current_time_slot, user_index, satellite_index] = 1
                else:
                    self.coverage_indicator[current_time_slot, user_index, satellite_index] = 0

    def calculate_actual_rate(self, time_slot: int, user_index: int, satellite_index: int) -> float:
        capacity = self.channel_capacity[satellite_index, user_index, time_slot]
        demand = self.user_demand_rate[user_index, time_slot]
        # 计算实际传输速率
        if demand <= capacity:
            return demand.item()
        else:
            return capacity.item()

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


if __name__ == "__main__":
    env = Env()
    # 打印动作空间和观察空间的形式
    print("Action Space:")
    print(env.action_space)

    print("\nObservation Space:")
    print(env.observation_space)

    # 打印观察空间中每个子空间的形式
    for key, space in env.observation_space.spaces.items():
        print(f"\nObservation Space - {key}:")
        print(space)
    # obs, info = env.reset()
    # done = False
    # while not done:
    #     action = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     env.render()
    #     done = terminated or truncated
    # env.close()
