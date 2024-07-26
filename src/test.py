import numpy as np
from env import Env  # 请确保将 your_env_module 替换为你实际的模块名


def test_env():
    # 实例化环境
    env = Env()

    # 重置环境，获取初始观察和信息
    obs, info = env.reset()
    print("Initial observation:", obs)
    print("Initial info:", info)

    done = False
    step_count = 0
    max_steps = 100  # 设定一个最大步数，以防止无限循环

    while not done and step_count < max_steps:
        # 从动作空间中采样一个随机动作
        action = env.action_space.sample()

        # 执行一步环境操作
        obs, reward, terminated, truncated, info = env.step(action)

        # 打印当前步的信息
        print(f"Step: {step_count}")
        print("Observation:", obs)
        print("Reward:", reward)
        print("Terminated:", terminated)
        print("Truncated:", truncated)
        print("Info:", info)

        # 渲染环境（可选）
        env.render()

        # 检查是否结束
        done = terminated or truncated
        step_count += 1

    # 关闭环境
    env.close()
    print("Test completed.")


if __name__ == "__main__":
    test_env()
