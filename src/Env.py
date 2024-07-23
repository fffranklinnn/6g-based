import numpy as np
import gym
import pandas as pd
from gym import spaces


class Env(gym.Env):
    def __init__(self):
        # 定义卫星和用户的数量
        self.NUM_SATELLITES = 300  # 卫星数量
        self.NUM_USERS = 20  # 用户数量
        self.NUM_GROUND_USER = 10  # 地面用户数量
        self.NUM_AERIAL_USER = self.NUM_USERS - self.NUM_GROUND_USER  # 空中用户数量，计算得出
        self.TOTAL_TIME = 3000  # 总模拟时间，单位：秒
        self.NUM_TIME_SLOTS = 60  # 时间段的划分数量
        self.TIME_SLOT_DURATION = self.TOTAL_TIME // self.NUM_TIME_SLOTS  # 每个时间段的持续时间

        # 初始化卫星和用户的位置
        self.satellite_positions = self.initialize_satellites()
        self.user_positions, self.user_elevations = self.initialize_ground_user()

        # 初始化覆盖指示变量和接入决策变量
        self.coverage_indicator = np.zeros((self.NUM_SATELLITES, self.NUM_USERS, self.TOTAL_TIME))
        self.access_decision = np.zeros((self.NUM_SATELLITES, self.NUM_USERS, self.TOTAL_TIME))

    def initialize_ground_user(self):
        #从data_ground.csv中读取数据，这个csv中总共有NUM_TIME_SLOTS行，每行的第一个数据为时间，从每行的第二个数据开始记录着每个地面用户的信息（包括经度，纬度和观察每个卫星的俯仰角（总共NUM_SATELLITES个角度））
        df = pd.read_csv('data_ground.csv')
        user_positions = np.zeros((self.NUM_GROUND_USER, self.NUM_TIME_SLOTS, 2))  # 经度和纬度
        user_elevations = np.zeros((self.NUM_GROUND_USER, self.NUM_TIME_SLOTS, self.NUM_SATELLITES))  # 每个卫星的仰角

        for i in range(self.NUM_TIME_SLOTS):
            time_data = df.iloc[i]
            for j in range(self.NUM_GROUND_USER):
                user_positions[j, i, 0] = time_data[1 + j * (2 + self.NUM_SATELLITES)]  # 经度
                user_positions[j, i, 1] = time_data[2 + j * (2 + self.NUM_SATELLITES)]  # 纬度
                user_elevations[j, i, :] = time_data[3 + j * (2 + self.NUM_SATELLITES):3 + (j + 1) * (
                            2 + self.NUM_SATELLITES)]  # 仰角

        return user_positions, user_elevations

    def initialize_satellites(self):
        #从data_satellite.csv中读取数据，这个csv中总共有NUM_TIME_SLOTS行，每行的第一个数据为时间，从每行的第二个数据开始记录着每个卫星的信息（包括经度，纬度和离地高度）
        df = pd.read_csv('data_satellite.csv')
        satellite_positions = np.zeros((self.NUM_SATELLITES, self.NUM_TIME_SLOTS, 3))  # 经度、纬度和离地高度

        for i in range(self.NUM_TIME_SLOTS):
            time_data = df.iloc[i]
            for j in range(self.NUM_SATELLITES):
                satellite_positions[j, i, 0] = time_data[1 + j * 3]  # 经度
                satellite_positions[j, i, 1] = time_data[2 + j * 3]  # 纬度
                satellite_positions[j, i, 2] = time_data[3 + j * 3]  # 离地高度

        return satellite_positions

    def initialize_aerial_user(self):
        #暂时不考虑
        pass

    def step(self,action):
        #暂时不考虑
        pass

    def reset(self):
        #暂时不考虑
        pass

    def calculate_snr(self, satellite_position, user_position):
        # 计算信噪比，这里简化为与距离的反比关系
        distance = np.linalg.norm(satellite_position - user_position)  # 计算欧氏距离
        snr = 1 / (distance + 1)  # 信噪比计算公式，加1避免除零错误
        return snr

    def calculate_interference(self, user_index, satellite_index, time, access_decision, num_users):
        # 计算给定用户和卫星在特定时间的干扰
        interference = 0
        for k in range(num_users):
            if k != user_index and access_decision[satellite_index, k, time] == 1:
                interference += 0.1  # 假设每个其他用户产生0.1的干扰
        return interference

    def simulate_time_slots(self, total_time, num_users, num_satellites, satellite_positions, user_positions, coverage_indicator, access_decision):
        # 模拟每个时间段的覆盖和接入决策
        for t in range(total_time):
            for k in range(num_users):
                for n in range(num_satellites):
                    # 计算覆盖指示变量
                    snr = self.calculate_snr(satellite_positions[n], user_positions[k])
                    if snr > 0.5:  # 如果信噪比大于0.5，则认为该用户被覆盖
                        coverage_indicator[n, k, t] = 1
                    else:
                        coverage_indicator[n, k, t] = 0

                    # 基于覆盖情况计算接入决策变量
                    if coverage_indicator[n, k, t] == 1:
                        interference = self.calculate_interference(k, n, t, access_decision, num_users)
                        if interference < 1:  # 如果干扰小于1，则允许接入
                            access_decision[n, k, t] = 1
                        else:
                            access_decision[n, k, t] = 0
                    else:
                        access_decision[n, k, t] = 0

    def run_simulation(self):
        # 运行模拟，模拟每个时间段的操作
        self.simulate_time_slots(self.TOTAL_TIME, self.NUM_USERS, self.NUM_SATELLITES, self.satellite_positions, self.user_positions, self.coverage_indicator, self.access_decision)

        # 输出结果
        print("覆盖指示变量 coverage_indicator:")
        print(self.coverage_indicator)
        print("接入决策变量 access_decision:")
        print(self.access_decision)

if __name__ == "__main__":
    env = Env()
    env.run_simulation()

    # def initialize_positions(self, num_satellites, num_users):
    #     # 随机生成卫星和用户的位置
    #     satellite_positions = np.random.rand(num_satellites, 2)  # 卫星位置 (x, y)
    #     user_positions = np.random.rand(num_users, 2)  # 用户位置 (x, y)
    #     return satellite_positions, user_positions

    # def initialize_variables(self, num_satellites, num_users, total_time):
    #     # 初始化覆盖指示变量和接入决策变量为全零矩阵
    #     coverage_indicator = np.zeros((num_satellites, num_users, total_time), dtype=int)
    #     access_decision = np.zeros((num_satellites, num_users, total_time), dtype=int)
    #     return coverage_indicator, access_decision