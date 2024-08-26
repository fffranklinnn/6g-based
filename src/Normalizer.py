import numpy as np


class Normalizer:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.mean = np.zeros(state_dim)
        self.var = np.ones(state_dim)
        self.count = 1e-4  # 避免除以零

    def update(self, state):
        self.count += 1
        self.mean = self.mean + (state - self.mean) / self.count
        self.var = self.var + (state - self.mean) ** 2 / self.count

    def normalize(self, state):
        return (state - self.mean) / (np.sqrt(self.var) + 1e-8)

    def denormalize(self, state):
        return state * (np.sqrt(self.var) + 1e-8) + self.mean


class ComplexNormalizer:
    def __init__(self, num_satellites, num_ground_user):
        self.coverage_space = Normalizer(num_satellites * num_ground_user * 2)
        self.previous_access_strategy_space = Normalizer(num_satellites * num_ground_user)
        self.switch_count_space = Normalizer(num_ground_user)
        self.elevation_angle_space = Normalizer(num_satellites * num_ground_user)
        self.altitude_space = Normalizer(num_satellites)

    def update(self, state):
        self.coverage_space.update(state[:self.coverage_space.state_dim])
        offset = self.coverage_space.state_dim

        self.previous_access_strategy_space.update(state[offset:offset + self.previous_access_strategy_space.state_dim])
        offset += self.previous_access_strategy_space.state_dim

        self.switch_count_space.update(state[offset:offset + self.switch_count_space.state_dim])
        offset += self.switch_count_space.state_dim

        self.elevation_angle_space.update(state[offset:offset + self.elevation_angle_space.state_dim])
        offset += self.elevation_angle_space.state_dim

        self.altitude_space.update(state[offset:offset + self.altitude_space.state_dim])

    def normalize(self, state):
        if state.ndim == 1:
            state = state[np.newaxis, :]  # 将一维数组转换为二维数组
        normalized_state = np.zeros_like(state)

        for i in range(state.shape[0]):
            normalized_state[i, :self.coverage_space.state_dim] = self.coverage_space.normalize(
                state[i, :self.coverage_space.state_dim])
            offset = self.coverage_space.state_dim

            normalized_state[i, offset:offset + self.previous_access_strategy_space.state_dim] = \
                self.previous_access_strategy_space.normalize(
                    state[i, offset:offset + self.previous_access_strategy_space.state_dim])
            offset += self.previous_access_strategy_space.state_dim

            normalized_state[i, offset:offset + self.switch_count_space.state_dim] = \
                self.switch_count_space.normalize(state[i, offset:offset + self.switch_count_space.state_dim])
            offset += self.switch_count_space.state_dim

            normalized_state[i, offset:offset + self.elevation_angle_space.state_dim] = \
                self.elevation_angle_space.normalize(state[i, offset:offset + self.elevation_angle_space.state_dim])
            offset += self.elevation_angle_space.state_dim

            normalized_state[i, offset:offset + self.altitude_space.state_dim] = \
                self.altitude_space.normalize(state[i, offset:offset + self.altitude_space.state_dim])

        return normalized_state

