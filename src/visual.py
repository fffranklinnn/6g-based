import numpy as np
import matplotlib.pyplot as plt

# 第一组奖励
rewards1 = [
    5.0288, 4.9237, 4.5469, 4.4221, 4.9423,
    6.4739, 12.2282, 29.0612, 35.9020, 49.3478,
    41.1035, 64.1536, 59.2316, 67.5836, 44.2527,
    59.5223, 47.2659, 60.7533, 41.8013, 36.0212,
    60.3487, 52.4737, 59.1735, 63.9679, 58.1872,
    69.9258, 62.9387, 35.7890, 55.5834, 49.9203,
    44.3085, 61.8502, 62.1592, 53.0324, 50.7806,
    33.2441, 57.7921, 56.4138, 76.5706, 57.2655,
    63.6865, 66.7245, 36.9656, 85.7780, 52.9096,
    40.7173, 65.7080, 60.1996, 59.7731, 47.5892
]

# 第二组奖励 - 填充相同长度
rewards2 = [
    4.6073, 4.5868, 4.4191, 4.3836, 5.1318,
    6.5175, 11.0427, 29.1274, 35.8723, 53.8332,
    48.3597, 73.1468, 58.2989, 60.7877, 62.9035,
    46.3989, 42.7086, 62.8248, 45.1958, 85.3975,
    67.2595, 49.0300, 49.4536, 59.0233, 68.0828,
    47.1688, 54.6501, 45.7099, 67.1864, 46.1921,
    51.3738, 59.3741, 57.1722, 69.4327, 57.3716,
    66.3598, 53.4823, 54.8551, 63.1159, 39.7694,
    58.8255, 51.5686, 51.2825, 61.7058, 40.3493,
    27.5530, 39.2815, 46.9799, 64.9173, 51.7115
]

# 第三组奖励 - 填充相同长度
rewards3 = [0.5658, -0.0655, 0.0639, 0.0302, 0.0971,
            0.4736, 1.2833, 1.7755, 0.2727, 2.2613,
            1.1016, 0.3068, -0.0395, 1.4715, 0.1022,
            4.7603, 2.2059, -0.1352, 4.9054, -0.1406,
            0.1958, 4.9057, 5.3311, 1.3656, 4.4556,
            3.3089, 0.1403, 1.2435, 0.0411, 3.9099,
            3.0797, 4.5401, -0.1109, 0.7961, 3.3527,
            7.4372, -0.0837, 7.1365, 2.7916, 3.5947,
            3.3117, -0.1383, 8.3685, 1.5541, 6.2863,
            10.0411, 8.2265, 6.5746, 0.2581, -0.0803]
rewards4 = [
    4.9575, 4.6634, 4.8815, 4.6681, 4.7492, 6.2746, 11.4186, 19.6800, 40.4620, 40.5009,
    64.7669, 55.0680, 46.0775, 85.8438, 78.2120, 84.9743, 113.2880, 93.5260, 55.6119, 74.0153,
    53.5889, 78.6568, 84.0533, 99.9484, 64.4250, 50.9515, 51.8407, 82.4048, 67.4754, 59.9504,
    106.5385, 61.4369, 88.0347, 74.8629, 83.3307, 96.7000, 95.4536, 70.4323, 51.4826, 71.8764,
    86.8554, 89.7164, 87.5447, 59.7406, 59.7370, 39.4979, 65.9130, 56.3437, 121.7957, 93.6210
]
# 画图
plt.figure(figsize=(12, 6))
plt.plot(rewards1, label='DQN', color='blue')
plt.plot(rewards2, label='SAC', color='green')
plt.plot(rewards3, label='RANDOM', color='red')
plt.plot(rewards4, label='DQN2', color='black')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards from Different Datasets')
plt.legend()
plt.show()
