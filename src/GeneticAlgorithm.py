import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

# 检查CUDA是否可用，如果可用则将设备设置为CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义基础变量
T = 60   # 定义时隙数量
sta_num = 301  # 定义卫星数量
client_num = 10  # 定义用户数量

# 定义场景参数
angle_threshold = 15  # 定义倾角阈值
# w1 = 0.4  # 定义切换次数权重
w1 = 5e-6
# w2 = 0.6# 定义用户速率权重
w2 = 4e-7
communication_frequency = torch.tensor(18.5e9) # 通信频率为18.5GHz
total_bandwidth = 250e4  # 总带宽为250MHz
noise_temperature = 213.15  # 系统的噪声温度为213.15开尔文
Polarization_isolation_factor = 12  # 单位dB
receive_benefit_ground = 15.4  # 单位dB
EIRP = 73.1  # 单位:dBm
k = 1.380649e-23  # 单位:J/K
radius_earth = 6731.0  # 单位:km
# EIRP_watts = 10 ** ((EIRP - 30) / 10)  # 将 EIRP 从 dBm 转换为瓦特
noise_power = k * noise_temperature * total_bandwidth  # 噪声功率计算

# 初始化种群，形状为60 * 10 * 300 ,约束条件也已满足
def initialize_population(pop_size, var_shape):
    population = []
    for _ in range(pop_size):
        individual = torch.zeros(var_shape, dtype=torch.float32, device=device)
        for t in range(var_shape[0]):
            available_satellites = torch.arange(var_shape[2], device=device)
            for k in range(var_shape[1]):
                if len(available_satellites) > 0:
                    chosen_satellite = torch.randint(len(available_satellites), (1,), device=device).item()
                    satellite_index = available_satellites[chosen_satellite]
                    individual[t, k, satellite_index] = torch.randint(0, 2, (1,), dtype=torch.float32).item()
                    available_satellites = torch.cat((
                        available_satellites[:chosen_satellite],
                        available_satellites[chosen_satellite+1:]
                    ))
        population.append(individual)
    return population


# 获取仰角数据,返回一个60*10*301的numpy数组
def gain_evl(T, client_num, sta_num):
    df = pd.read_csv('ev_data.csv')
    var_shape = (T, client_num, sta_num)
    evl = torch.zeros(var_shape, dtype=torch.float32, device=device)

    for t in range(T):
        for n in range(client_num):
            for k in range(sta_num):
                evl[t, n, k] = df.iloc[k * client_num + n, t]
    return evl
#获取覆盖性指标
def initialize_coverage(T, client_num, sta_num):
    csv_file = 'coverage_data.csv'

    df = pd.read_csv(csv_file, header=None, skiprows=1)

    # print(f"cov DataFrame shape: {df.shape}")

    # 从CSV文件读取覆盖数据
    coverage = torch.zeros((T, client_num, sta_num), dtype=torch.float32)

    # 填充 coverage 数组
    for time_slot in range(T):
        for i in range(sta_num):
            for j in range(client_num):
                # 获取覆盖字符串并解析成整数
                coverage_str = df.iloc[j + i * client_num, time_slot]
                beam_1, beam_2 = map(int, coverage_str.strip('()').split(','))
                if beam_1 == 1 or beam_2 == 1:
                    coverage[time_slot, j, i] = 1
                else:
                    coverage[time_slot, j, i] = 0


        # print(f"Initialized coverage with shape: {coverage.shape}")
    return coverage.to(device)
coverage = initialize_coverage(T, client_num, sta_num)
# 计算倾角覆盖矩阵 分析了一下逻辑 感觉这个没有必要了
'''
def calculate_coverge(T, client_num, sta_num):
    evl = gain_evl(T, client_num, sta_num)
    var_shape = (T, client_num, sta_num)
    coverage_indicator = torch.zeros(var_shape, dtype=torch.float32, device=device)
    for time_slot in range(T):
        for user_index in range(client_num):
            for satellite_index in range(sta_num):
                if evl[time_slot, user_index, satellite_index] > angle_threshold:
                    coverage_indicator[time_slot, user_index, satellite_index] = 1
                else:
                    coverage_indicator[time_slot, user_index, satellite_index] = 0
    return coverage_indicator

coverage_indicator_ind = calculate_coverge(T, client_num, sta_num)
'''
# 计算转换次数 已验证其正确性
def calculate_hk(individual):
    individual = individual.mul(coverage)
    # 在第一个尺度下计算元素变换的次数
    axis0_changes = torch.nonzero(torch.diff(individual, dim=0)).size(0)

    return axis0_changes

# 获取卫星高度
def gain_alt(T, sta_num):
    df = pd.read_csv('alt_data.csv')
    var_shape = (T, sta_num)
    alt = torch.zeros(var_shape, dtype=torch.float32, device=device)

    for t in range(T):
        for k in range(sta_num):
            # 确保从 DataFrame 中读取的数据是数值类型
            value = float(df.iloc[k, t])
            alt[t, k] = int(value)

    return alt

# 计算距离
def calculate_distance_matrix(T, client_num, sta_num)-> torch.Tensor :
    # 获取所有时间段的卫星高度和仰角
    sat_heights = gain_alt(T, sta_num)  # 形状: [61, 301]
    eval_angles = gain_evl(T, client_num, sta_num)  # 形状: [61, 10, 301]

    # 通过调整形状来启用广播
    sat_heights_expanded = torch.unsqueeze(sat_heights.clone().detach(), dim=1)
    # 注意：这里不再对 eval_angles 进行形状调整，因为它已经是预期形状


    # 计算距离公式
    distance = radius_earth * (radius_earth + sat_heights_expanded) / torch.sqrt(
        (radius_earth + sat_heights_expanded) ** 2 - radius_earth ** 2 * torch.cos(torch.deg2rad(eval_angles)))

        # print(f"[calculate_distance_matrix] Distance matrix shape: {distance.shape}")
        # 断言验证最终形状
        # assert distance.shape == (61, 10, 301), f"Unexpected shape: {distance.shape}"

    return distance

distance = calculate_distance_matrix(T, client_num, sta_num)
def calculate_DL_pathloss_matrix(distance_matrix)-> torch.Tensor:
    # 计算路径损耗矩阵
    pathloss = 20 * torch.log10(distance_matrix*1e3) + 20 * torch.log10(communication_frequency.clone().detach()) - 147.55

    # print(f"Pathloss matrix shape: {pathloss.shape}")
    return pathloss

#CNR的计算需要根据决策变量来决定，所以应该只记录当前slot下的CNR情况
def calculate_CNR_matrix(individual,distance_matrix)-> torch.Tensor :
    # 计算路径损耗矩阵，其形状为 [NUM_TIME_SLOTS, NUM_SATELLITES, NUM_GROUND_USER]
    loss = calculate_DL_pathloss_matrix(distance_matrix)

    # 计算接收功率（单位：瓦特），假设 self.EIRP_watts 和 self.receive_benefit_ground 是标量
    # received_power_watts = EIRP_watts * 10 ** (receive_benefit_ground / 10) / (10 ** (loss / 10))

    CNR = EIRP +receive_benefit_ground -30 - 10 * torch.log10(k * total_bandwidth * noise_temperature * loss)
    return CNR
    # print(f"received power watts:",{received_power_watts})

    # 计算 CNR（线性值），假设 self.noise_power 是标量
    # CNR_linear = received_power_watts / noise_power
        # print(f"CNR Linear:",{CNR_linear})
        # 返回 CNR 的对数值（单位：dB），保持矩阵形状
    # CNR_linear = 10 * torch.log10(CNR_linear)
        # print(f"CNR:",{CNR})
        # print(f"[calculate_CNR_matrix] CNR matrix shape: {CNR.shape}")  # [10,301,301]
    # CNR_linear2 = CNR_linear.mul(coverage)
    #print('CNR_linear:')
    #print(CNR_linear)
    # CNR_linear3 = CNR_linear.mul(individual)
    # return CNR_linear3

# def calculate_interference_matrix(individual,distance_matrix)-> torch.Tensor :
#     CNR = calculate_CNR_matrix(individual,distance_matrix)
#     for t in range(T):
#         row_sums =

def calculate_interference_matrix(individual,T, client_num, sta_num)-> torch.Tensor:
    CNR_linear2 = calculate_CNR_matrix(individual, distance)
    interference_matrix = torch.zeros((T,client_num, sta_num), dtype=torch.float32, device=device)
    for t in range(T):
        row_sums = CNR_linear2[t].sum(dim=1)
        #print(row_sums)
        row_sums_expanded = row_sums.unsqueeze(1).expand_as(CNR_linear2[t])
        result2 = (row_sums_expanded - CNR_linear2[t]).mul(individual[t])
        #print(result2)
        interference_matrix[t] = result2
    return interference_matrix
def update_rates_and_capacity(individual):
    CNR_linear2 = calculate_CNR_matrix(individual, distance)
    #print('CNR_linear2:')
    #print(CNR_linear2)
    interference_matrix = calculate_interference_matrix(individual,T, client_num, sta_num)
    #print('interference_matrix:')
    #print(interference_matrix)
    channel_capacity = total_bandwidth * torch.log2(1.0 + CNR_linear2 / (interference_matrix + 1.0))
    return channel_capacity

# 计算信道容量
def calculate_R(individual, t):
    reward = 0
    channel_capacity = update_rates_and_capacity(individual)
    #print('channel_capacity:')
    #print(channel_capacity)
    # 找到张量中非零元素的索引
    nonzero_indiceschannel_capacity = torch.nonzero(channel_capacity)

    # 提取出非零元素
    #nonzero_elementschannel_capacity = channel_capacity[nonzero_indiceschannel_capacity[:, 0], nonzero_indiceschannel_capacity[:, 1]]
    #print(nonzero_elementschannel_capacity)
    # 对应元素相乘
    result = torch.mul(individual[t], channel_capacity[t])
    #print(result)
    # print(result)
    # 找到张量中非零元素的索引
    nonzero_indices = torch.nonzero(result)

    # 提取出非零元素
    nonzero_elements = result[nonzero_indices[:, 0], nonzero_indices[:, 1]]
    #print('nonzero_elements:')
    #print(nonzero_elements)
    # 对非零元素进行相加
    capacity = torch.sum(nonzero_elements)

    reward += capacity
    #print('reward:')
    #print(reward)
    return reward

# 定义适应度函数，即就是优化目标
def calculate_fitness(individual, T):

    hk = -calculate_hk(individual)
    r = 0
    for t in range(T):
        rt = calculate_R(individual, t)
        r += rt
    fitness = w1*hk + w2*r
    return fitness

def select_parents(population, fitness):
    fitness_tensor = torch.tensor(fitness, device='cuda')
    # 选择适应度最高的两个个体的索引
    # indices = torch.argsort(fitness_tensor)[-2:]

    # 选择适应度最低的两个个体的索引
    indices = torch.argsort(fitness_tensor)[-2:]
    return [population[i] for i in indices]

def crossover(parent1, parent2):
    point1 = torch.randint(1, parent1.shape[2] - 2, (1,), device='cuda')
    point2 = torch.randint(point1 + 1, parent1.shape[2] - 1, (1,), device='cuda')
    child1 = torch.cat((parent1[:, :, :point1], parent2[:, :, point1:point2], parent1[:, :, point2:]), dim=2)
    child2 = torch.cat((parent2[:, :, :point1], parent1[:, :, point1:point2], parent2[:, :, point2:]), dim=2)
    return child1, child2

def mutate(individual, generation, max_generations, mutation_rate=0.1):
    for t in range(individual.shape[2]):
        for k in range(individual.shape[1]):
            if torch.rand(1, device='cuda') < mutation_rate * (1 - generation / max_generations):
                n = torch.randint(individual.shape[0], (1,), device='cuda')
                individual[:, k, t] = 0
                individual[n, k, t] = 1
    return individual


# Genetic Algorithm Parameters
pop_size = 50
max_generations = 100
mutation_rate = 0.05

# Initialize population
population = initialize_population(pop_size, (T, client_num, sta_num))
# Initialize best fitness and fitness history list
#best_fitness = float('inf')
best_fitness = 0.0
fitness_history = []
# Run Genetic Algorithm
for generation in range(max_generations):
    fitness = [calculate_fitness(ind, T) for ind in population]
    parents = select_parents(population, fitness)

    new_population = []
    for _ in range(pop_size // 2):
        child1, child2 = crossover(parents[0], parents[1])
        new_population.append(mutate(child1, generation, max_generations, mutation_rate))
        new_population.append(mutate(child2, generation, max_generations, mutation_rate))

    population = new_population
    current_best_fitness = max(fitness)
    current_best_fitness = current_best_fitness
    if current_best_fitness > best_fitness:
        best_fitness = current_best_fitness
    fitness_history.append(best_fitness)
    print(f"Generation {generation}: Best Fitness = {best_fitness}")


fitness_history = [x.cpu().numpy() for x in fitness_history]

#print(fitness_history)
# Plot fitness history
plt.plot(fitness_history)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.show()