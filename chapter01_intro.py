"""第 1 章 初探强化学习 —— 补充示例

说明：书中第 1 章是概念引入（智能体与环境的交互、回报、探索与利用），
原文没有附带代码。这里补一个最小的 gymnasium 示例，演示后续各章都会用到的
"环境接口" 与 "回报计算"，方便先跑通再说：

    1) gymnasium 的标准交互循环：reset -> step -> (obs, reward, terminated, truncated, info)
    2) 用折扣回报 G_t = Σ γ^k r_{t+k} 评估一条轨迹
    3) 用 ansi 渲染看一眼网格世界（第 4~6 章的悬崖漫步）

运行：
    python chapter01_intro.py
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from hrl.envs import make_env


def random_rollout(env_name: str = "CartPole-v1", gamma: float = 0.99,
                   seed: int = 0) -> None:
    """用随机策略跑一个回合，打印交互过程和折扣回报。"""
    env = gym.make(env_name)
    obs, info = env.reset(seed=seed)  # 新版接口：reset 返回 (obs, info)
    print(f"环境: {env_name}")
    print(f"  观测空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    print(f"  初始观测: {np.round(obs, 3)}")

    rewards, done, t = [], False, 0
    while not done:
        action = env.action_space.sample()  # 随机策略：这里只为演示接口
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
        t += 1
        if t <= 3 or done:
            print(f"  t={t:<3} action={action} reward={reward} "
                  f"terminated={terminated} truncated={truncated}")
    env.close()

    # 折扣回报：从最后一个奖励往前递推
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
    print(f"  共 {len(rewards)} 步，折扣回报 G_0 = {G:.3f}\n")


def show_grid_world() -> None:
    """用 ansi 渲染第 4~6 章会用到的悬崖漫步环境。"""
    env = make_env("book", render_mode="ansi")
    state, info = env.reset(seed=0)
    print("悬崖漫步初始状态（x=智能体，C=悬崖，T=终点）：")
    print(env.render(), end="")
    print(f"初始状态编号: {state}\n")

    # 向下走进悬崖：reward=-100，terminated=True
    next_state, reward, terminated, truncated, info = env.step(1)
    print(f"在起点执行'下' -> state={next_state}, reward={reward}, "
          f"terminated={terminated}")
    env.close()


def main() -> None:
    random_rollout()
    show_grid_world()


if __name__ == "__main__":
    main()
