import numpy as np
import gym
import pandas as pd
from gym import spaces
from gym.spaces import Box, Discrete, MultiBinary


class Env(gym.Env):
    def __init__(self):
        # 定义卫星和用户的数量
        self.NUM_SATELLITES = 300  # 卫星数量
        # self.NUM_USERS = 20  # 用户数量
        self.NUM_GROUND_USER = 10  # 地面用户数量
        # self.NUM_AERIAL_USER = self.NUM_USERS - self.NUM_GROUND_USER  # 空中用户数量，计算得出
        self.TOTAL_TIME = 3000  # 总模拟时间，单位：秒
        self.NUM_TIME_SLOTS = 60  # 时间段的划分数量
        self.TIME_SLOT_DURATION = self.TOTAL_TIME // self.NUM_TIME_SLOTS  # 每个时间段的持续时间
        self.communication_frequency = 18.5e9  # 通信频率为18.5GHz
        self.total_bandwidth = 250e6  # 总带宽为250MHz
        self.noise_temperature = 213.15  # 系统的噪声温度为213.45~273.15开尔文
        self.Polarization_isolation_factor = 12  # 单位dB
        self.receive_benefit_ground = 15.4  # 单位dB
        self.EIRP = 73.1  # 单位:dBm
        self.k = 1.380649e-23  # 单位:J/K
        self.radius_earth = 6731e3  # 单位:m
        self.action_space = MultiBinary(self.NUM_GROUND_USER * self.NUM_SATELLITES)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # 初始化卫星和用户的位置
        self.satellite_heights = self.initialize_satellite()
        self.eval_angle = self.initialize_ground()
        self.angle_threshold = 15  # 单位：度

        # 初始化覆盖指示变量和接入决策变量
        self.coverage_indicator = np.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER, self.TOTAL_TIME))
        self.access_decision = np.zeros((self.NUM_SATELLITES, self.NUM_GROUND_USER, self.TOTAL_TIME))
        self.current_time_step = 0

    def initialize_ground(self):
        # 从data_ground.csv中读取数据，这个csv中总共有NUM_TIME_SLOTS行，
        # 每行第一个数据为时间，从第二个数据开始记录每个地面用户到每个卫星的俯仰角
        # 俯仰角要用数组eval_angle[][][]记录，其中第一个维度是时间，第二三个维度分别为地面用户的编号和卫星的编号，
        # 例如eval[1][2][3]表示t=1时用户编号为2的地面用户与卫星编号为3的卫星的俯仰角
        df = pd.read_csv('data_ground.csv')
        # 假设CSV文件的列数是 NUM_GROUND_USER * NUM_SATELLITES + 1（时间列）
        eval_angle = np.zeros((self.NUM_TIME_SLOTS, self.NUM_GROUND_USER, self.NUM_SATELLITES))

        for index, row in df.iterrows():
            time_slot = int(row[0])
            for i in range(self.NUM_GROUND_USER):
                for j in range(self.NUM_SATELLITES):
                    eval_angle[time_slot, i, j] = row[i * self.NUM_SATELLITES + j + 1]

        return eval_angle

    def initialize_satellite(self):
        # 从data_satellite.csv中读取数据，这个csv中总共有NUM_TIME_SLOTS行，
        # 每行第一个数据为时间，从第二个数据开始记录每个卫星的高度（距离地表，单位m）
        # 返回一个数组sat_heights[][]，第一个维度是时间，第二个纬度是卫星编号，例如sat_heights[1][4]表示t为1时卫星编号为4的卫星的离地高度
        df = pd.read_csv('data_satellite.csv')
        sat_heights = np.zeros((self.NUM_TIME_SLOTS, self.NUM_SATELLITES))

        for index, row in df.iterrows():
            # 假设第一个数据是时间槽索引，从0开始
            time_slot = int(row[0])
            # 从第二个数据开始是各个卫星的高度
            for i in range(self.NUM_SATELLITES):
                sat_heights[time_slot, i] = row[i + 1]

        return sat_heights

    def step(self, action):
        # 当前时隙索引，假设您有一个变量来跟踪当前的时间槽
        current_time_slot = self.get_current_time_slot()

        # 在每个时隙开始时更新覆盖情况
        self.update_coverage_indicator(current_time_slot)

        # 将一维动作数组转换为二维形式
        action_matrix = action.reshape((self.NUM_GROUND_USER, self.NUM_SATELLITES))

        # 根据action_matrix更新环境状态
        # 注意：这里需要添加你的环境逻辑，例如更新用户卫星连接状态等

        # 计算奖励
        reward = self.calculate_reward(action_matrix)

        # 检查是否结束
        done = self.check_if_done()

        # 可选的额外信息
        info = {}

        return self.get_observation(), reward, done, info

    def reset(self):
        #暂时不考虑
        pass

    def get_observation(self):
        # 获取当前环境的观察
        observation = np.random.random(self.NUM_SATELLITES + self.NUM_GROUND_USER)  # 示例：随机生成观察
        return observation

    def calculate_reward(self, action_matrix):
        # 根据动作和当前状态计算奖励
        reward = 0
        # 奖励计算逻辑...
        return reward

    def calculate_distance(self, time_slot, user_index, satellite_index):
        # 使用指定时间槽的卫星高度和俯仰角
        sat_height = self.satellite_heights[time_slot, satellite_index]
        eval_angle = self.eval_angle[time_slot, user_index, satellite_index]
        # 计算距离
        result = self.radius_earth * (self.radius_earth + sat_height) / np.sqrt(
            np.pow((self.radius_earth + sat_height), 2) - self.radius_earth ** 2 * np.cos(np.radians(eval_angle)) ** 2)
        return result

    def calculate_DL_pathloss(self, time_slot, user_index, satellite_index):
        distance = self.calculate_distance(time_slot, user_index, satellite_index)
        result = 20 * np.log10(distance) + 20 * np.log10(self.communication_frequency) - 147.55
        return result

    def calculate_CNR(self, time_slot, user_index, satellite_index):
        loss = self.calculate_DL_pathloss(time_slot, user_index, satellite_index)
        EIRP_watts = 10 ** ((self.EIRP - 30) / 10)  # 将 EIRP 从 dBm 转换为瓦特
        noise_power = self.k * self.noise_temperature * self.total_bandwidth  # 噪声功率计算
        received_power_watts = EIRP_watts * 10 ** (self.receive_benefit_ground / 10) / (10 ** (loss / 10))  # 接收功率
        CNR_linear = received_power_watts / noise_power  # 线性单位的载噪比
        result = 10 * np.log10(CNR_linear)  # 转换为 dB
        return result

    def calculate_interference(self, time_slot, user_index, accessed_satellite_index):
        # 初始化总干扰功率为0
        total_interference_power_watts = 0

        # 遍历所有卫星，计算每个卫星对用户的干扰
        for satellite_index in range(self.NUM_SATELLITES):
            # 检查卫星是否不是当前接入的卫星并且在用户的覆盖范围内
            if satellite_index != accessed_satellite_index and self.coverage_indicator[
                satellite_index, user_index, time_slot] == 1:
                # 计算从该卫星到用户的下行路径损耗
                loss = self.calculate_DL_pathloss(time_slot, user_index, satellite_index)
                # 将 EIRP 从 dBm 转换为瓦特，以便进行线性计算
                EIRP_watts = 10 ** ((self.EIRP - 30) / 10)
                # 计算该卫星产生的干扰功率
                # 这里假设接收增益 self.receive_benefit_ground 已经是线性单位，如果是 dB，则需要转换
                interference_power_watts = EIRP_watts * (10 ** (self.receive_benefit_ground / 10)) / (10 ** (loss / 10))
                # 累加总干扰功率
                total_interference_power_watts += interference_power_watts

        # 将总干扰功率从瓦特转换为 dBm，以便与其他以 dBm 为单位的参数进行比较
        # 注意：转换回 dBm 需要加上 30 dBm (因为 1W = 30 dBm)
        total_interference_dBm = 10 * np.log10(total_interference_power_watts) + 30

        # 返回总干扰功率的 dBm 值
        return total_interference_dBm

    def update_coverage_indicator(self, current_time_slot):
        for user_index in range(self.NUM_GROUND_USER):
            for satellite_index in range(self.NUM_SATELLITES):
                # 检查俯仰角是否大于限定值
                if self.eval_angle[current_time_slot, user_index, satellite_index] > self.angle_threshold:
                    self.coverage_indicator[satellite_index, user_index, current_time_slot] = 1
                else:
                    self.coverage_indicator[satellite_index, user_index, current_time_slot] = 0

    # def update_coverage_indicator(self):
    #     for time_slot in range(self.NUM_TIME_SLOTS):
    #         for user_index in range(self.NUM_GROUND_USER):
    #             for satellite_index in range(self.NUM_SATELLITES):
    #                 # 检查俯仰角是否大于限定值
    #                 if self.eval_angle[time_slot, user_index, satellite_index] > self.angle_threshold:
    #                     self.coverage_indicator[satellite_index, user_index, time_slot] = 1
    #                 else:
    #                     self.coverage_indicator[satellite_index, user_index, time_slot] = 0


if __name__ == "__main__":
    env = Env()
    # env.run_simulation()


    # def simulate_time_slots(self, total_time, num_users, num_satellites, coverage_indicator, access_decision):
    #     # 模拟每个时间段的覆盖和接入决策
    #     for t in range(total_time):
    #         for k in range(num_users):
    #             for n in range(num_satellites):
    #                 # 计算覆盖指示变量,这里的逻辑应该是根据用户k到卫星n的俯仰角判断，如果小于threshold则为0，否则为1
    #
    #                 # 基于覆盖情况计算接入决策变量
    #                 if coverage_indicator[n, k, t] == 1:
    #                     interference = self.calculate_interference(k, n, t, access_decision, num_users)
    #                     if interference < 1:  # 如果干扰小于1，则允许接入
    #                         access_decision[n, k, t] = 1
    #                     else:
    #                         access_decision[n, k, t] = 0
    #                 else:
    #                     access_decision[n, k, t] = 0
    # def calculate_interference(self, user_index, satellite_index, time, access_decision, num_users):
    #     # 计算给定用户和卫星在特定时间的干扰
    #     interference = 0
    #     for k in range(num_users):
    #         if k != user_index and access_decision[satellite_index, k, time] == 1:
    #             interference += 0.1  # 假设每个其他用户产生0.1的干扰
    #     return interference
    # def calculate_distance(self, i, j):
    #     result = self.radius_earth * (self.radius_earth + self.satellite_heights[j]) / np.sqrt(
    #         np.pow((self.radius_earth + self.satellite_heights[j]), 2) - self.radius_earth ** 2 * self.eval_angle[i][j])
    #     return result
    # def calculate_DL_pathloss(self, i, j):
    #     distance = self.calculate_distance(i, j)
    #     result = 20 * np.log10(distance) + 20 * np.log10(self.communication_frequency) - 147.55
    #     return result
    # def calculate_CNR(self, i, j):
    #     loss = self.calculate_DL_pathloss(i, j)
    #     result = 10 * np.log10(
    #         self.EIRP * self.receive_benefit_ground / (self.k * self.noise_temperature * self.total_bandwidth * loss))
    #     return result
    # def calculate_snr(self, satellite_position, user_position):
    #     # 计算信噪比，这里简化为与距离的反比关系
    #     distance = np.linalg.norm(satellite_position - user_position)  # 计算欧氏距离
    #     snr = 1 / (distance + 1)  # 信噪比计算公式，加1避免除零错误
    #     return snr
    # def run_simulation(self):
    #     # 运行模拟，模拟每个时间段的操作
    #     self.simulate_time_slots(self.TOTAL_TIME, self.NUM_GROUND_USER, self.NUM_SATELLITES,
    #     self.coverage_indicator, self.access_decision)
    #
    #     # 输出结果
    #     print("覆盖指示变量 coverage_indicator:")
    #     print(self.coverage_indicator)
    #     print("接入决策变量 access_decision:")
    #     print(self.access_decision)
    # def initialize_ground_user(self):
    #     #从data_ground.csv中读取数据，这个csv中总共有NUM_TIME_SLOTS行，每行的第一个数据为时间，
    #     从每行的第二个数据开始记录着每个地面用户的信息（包括经度，纬度和观察每个卫星的俯仰角（总共NUM_SATELLITES个角度））
    #     df = pd.read_csv('data_ground.csv')
    #     user_positions = np.zeros((self.NUM_GROUND_USER, self.NUM_TIME_SLOTS, 2))  # 经度和纬度
    #     user_elevations = np.zeros((self.NUM_GROUND_USER, self.NUM_TIME_SLOTS, self.NUM_SATELLITES))  # 每个卫星的仰角
    #
    #     for i in range(self.NUM_TIME_SLOTS):
    #         time_data = df.iloc[i]
    #         for j in range(self.NUM_GROUND_USER):
    #             user_positions[j, i, 0] = time_data[1 + j * (2 + self.NUM_SATELLITES)]  # 经度
    #             user_positions[j, i, 1] = time_data[2 + j * (2 + self.NUM_SATELLITES)]  # 纬度
    #             user_elevations[j, i, :] = time_data[3 + j * (2 + self.NUM_SATELLITES):3 + (j + 1) * (
    #                         2 + self.NUM_SATELLITES)]  # 仰角
    #
    #     return user_positions, user_elevations

    # def initialize_satellites(self):
    #     #从data_satellite.csv中读取数据，这个csv中总共有NUM_TIME_SLOTS行，
    #     每行的第一个数据为时间，从每行的第二个数据开始记录着每个卫星的信息（包括经度，纬度和离地高度）
    #     df = pd.read_csv('data_satellite.csv')
    #     satellite_positions = np.zeros((self.NUM_SATELLITES, self.NUM_TIME_SLOTS, 3))  # 经度、纬度和离地高度
    #
    #     for i in range(self.NUM_TIME_SLOTS):
    #         time_data = df.iloc[i]
    #         for j in range(self.NUM_SATELLITES):
    #             satellite_positions[j, i, 0] = time_data[1 + j * 3]  # 经度
    #             satellite_positions[j, i, 1] = time_data[2 + j * 3]  # 纬度
    #             satellite_positions[j, i, 2] = time_data[3 + j * 3]  # 离地高度
    #
    #     return satellite_positions

    # def initialize_aerial_user(self):
    #     #暂时不考虑
    #     pass
