# data_test.py

import numpy as np
from env_gym import Env


def test_initialize_ground():
    env = Env()
    eval_angle = env.initialize_angle()

    # 检查 eval_angle 的形状是否正确
    assert eval_angle.shape == (env.NUM_TIME_SLOTS, env.NUM_GROUND_USER, env.NUM_SATELLITES), \
        "eval_angle 的形状不正确"

    # 打印 eval_angle 以进行手动检查
    print("Eval Angle:")
    print(eval_angle)

    # 检查 eval_angle 中的数据是否合理（根据实际数据进行调整）
    # 例如，检查某些值是否在预期范围内
    assert np.all((eval_angle >= -90) & (eval_angle <= 90)), \
        "eval_angle 中的值不在预期范围内"


def test_initialize_satellite():
    env = Env()
    sat_heights = env.initialize_altitude()

    # 检查 sat_heights 的形状是否正确
    assert sat_heights.shape == (env.NUM_TIME_SLOTS, env.NUM_SATELLITES), \
        "sat_heights 的形状不正确"

    # 打印 sat_heights 以进行手动检查
    print("Satellite Heights:")
    print(sat_heights)

    # 检查 sat_heights 中的数据是否合理（根据实际数据进行调整）
    # 例如，检查某些值是否在预期范围内
    assert np.all(sat_heights >= 0), \
        "sat_heights 中的值不在预期范围内"


if __name__ == "__main__":
    test_initialize_ground()
    test_initialize_satellite()
    print("所有测试通过！")
