"""第 3 章 马尔可夫决策过程（MDP）

对应书中示例：
  1) 给定状态序列，用定义计算回报 G_t
  2) 用贝尔曼方程的矩阵形式求 MRP / MDP 的解析解
  3) 按策略采样轨迹、用蒙特卡洛方法估计状态价值
  4) 用采样估计策略的占用度量（occupancy measure）

本章示例只做表格/概率运算，不需要 gymnasium 环境。
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- 1. 回报
np.random.seed(0)

# 定义状态转移概率矩阵 P（6 个状态的 MRP）
P_mrp = np.array([
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
])
rewards = np.array([-1, -2, -2, 10, 1, 0])  # 奖励函数
gamma = 0.5  # 折扣因子


def compute_return(start_index: int, chain, gamma: float) -> float:
    """给定一条序列，计算从某个索引（起始状态）开始到序列最后的回报。"""
    G = 0.0
    for i in reversed(range(start_index, len(chain))):
        G = gamma * G + rewards[chain[i] - 1]
    return G


def demo_return() -> None:
    chain = [1, 2, 3, 6]  # 一个状态序列 s1-s2-s3-s6
    G = compute_return(0, chain, gamma)
    print("根据本序列计算得到回报为：%s。" % G)


# ----------------------------------------------------------- 2. 解析解（贝尔曼方程）
def compute(P, rewards, gamma: float, states_num: int) -> np.ndarray:
    """利用贝尔曼方程的矩阵形式计算解析解，states_num 是状态数。"""
    rewards = np.array(rewards).reshape((-1, 1))  # 将 rewards 写成列向量
    value = np.dot(
        np.linalg.inv(np.eye(states_num, states_num) - gamma * P), rewards
    )
    return value


def demo_analytic() -> None:
    V = compute(P_mrp, rewards, gamma, 6)
    print("MRP 中每个状态价值分别为\n", V)


# ------------------------------------------------------- 3. MDP 的定义与策略
S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合
A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]  # 动作集合

# 状态转移函数 P(s'|s,a)
P_mdp = {
    "s1-保持s1-s1": 1.0,
    "s1-前往s2-s2": 1.0,
    "s2-前往s1-s1": 1.0,
    "s2-前往s3-s3": 1.0,
    "s3-前往s4-s4": 1.0,
    "s3-前往s5-s5": 1.0,
    "s4-前往s5-s5": 1.0,
    "s4-概率前往-s2": 0.2,
    "s4-概率前往-s3": 0.4,
    "s4-概率前往-s4": 0.4,
}

# 奖励函数 R(s,a)
R = {
    "s1-保持s1": -1,
    "s1-前往s2": 0,
    "s2-前往s1": -1,
    "s2-前往s3": -2,
    "s3-前往s4": -2,
    "s3-前往s5": 0,
    "s4-前往s5": 10,
    "s4-概率前往": 1,
}
gamma = 0.5  # 折扣因子
MDP = (S, A, P_mdp, R, gamma)

# 策略 1：随机策略
Pi_1 = {
    "s1-保持s1": 0.5,
    "s1-前往s2": 0.5,
    "s2-前往s1": 0.5,
    "s2-前往s3": 0.5,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.5,
    "s4-概率前往": 0.5,
}
# 策略 2
Pi_2 = {
    "s1-保持s1": 0.6,
    "s1-前往s2": 0.4,
    "s2-前往s1": 0.3,
    "s2-前往s3": 0.7,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.1,
    "s4-概率前往": 0.9,
}


def join(str1: str, str2: str) -> str:
    """把两个字符串用 '-' 连接，便于使用上面定义的 P、R 字典。"""
    return str1 + "-" + str2


def demo_analytic_mdp() -> None:
    """把策略 π 代入 MDP，得到 MRP 后求解析解。"""
    P_from_mdp_to_mrp = np.array([
        [0.5, 0.5, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.5],
        [0.0, 0.1, 0.2, 0.2, 0.5],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    R_from_mdp_to_mrp = [-0.5, -1.5, -1.0, 5.5, 0]
    V = compute(P_from_mdp_to_mrp, R_from_mdp_to_mrp, gamma, 5)
    print("MDP 中每个状态价值分别为\n", V)


# ------------------------------------------------------------------- 4. 采样轨迹
def sample(MDP, Pi, timestep_max: int, number: int):
    """采样函数：策略 Pi，限制最长步数 timestep_max，共采样 number 条序列。"""
    S, A, P, R, gamma = MDP
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)]  # 随机选择一个除 s5 以外的状态作为起点
        # 当前状态为终止状态或者时间步太长时，一次采样结束
        while s != "s5" and timestep <= timestep_max:
            timestep += 1
            a = _sample_action(Pi, s)
            r = R.get(join(s, a), 0)
            s_next = _sample_next_state(P, s, a, S)
            episode.append((s, a, r, s_next))  # 把 (s,a,r,s_next) 放入序列
            s = s_next
        episodes.append(episode)
    return episodes


def _sample_action(Pi, s):
    """按策略 Pi 在状态 s 下采样动作。"""
    rand, temp = np.random.rand(), 0.0
    for a_opt in A:
        temp += Pi.get(join(s, a_opt), 0)
        if temp > rand:
            return a_opt
    return A[-1]  # 数值误差兜底


def _sample_next_state(P, s, a, S):
    """按状态转移概率采样下一个状态。"""
    rand, temp = np.random.rand(), 0.0
    for s_opt in S:
        temp += P.get(join(join(s, a), s_opt), 0)
        if temp > rand:
            return s_opt
    return S[-1]  # 数值误差兜底


def demo_sample() -> None:
    episodes = sample(MDP, Pi_1, 20, 5)
    print("第一条序列\n", episodes[0])
    print("第二条序列\n", episodes[1])
    print("第五条序列\n", episodes[4])


# ------------------------------------------------------ 5. 蒙特卡洛估计状态价值
def MC(episodes, V, N, gamma: float) -> None:
    """对所有采样序列计算所有状态的价值（增量式求平均）。"""
    for episode in episodes:
        G = 0.0
        for i in range(len(episode) - 1, -1, -1):  # 从序列末尾往前算
            s, a, r, s_next = episode[i]
            G = r + gamma * G
            N[s] += 1
            V[s] += (G - V[s]) / N[s]


def demo_mc() -> None:
    episodes = sample(MDP, Pi_1, 20, 1000)  # 采样 1000 条，每条最长 20 步
    V = {s: 0.0 for s in S}
    N = {s: 0 for s in S}
    MC(episodes, V, N, gamma)
    print("使用蒙特卡洛方法计算 MDP 的状态价值为\n", V)


# --------------------------------------------------------------- 6. 占用度量
def occupancy(episodes, s, a, timestep_max: int, gamma: float) -> float:
    """计算状态动作对 (s,a) 出现的频率，以此估算策略的占用度量。"""
    rho = 0.0
    total_times = np.zeros(timestep_max)  # 每个时间步 t 被经历过几次
    occur_times = np.zeros(timestep_max)  # (s_t,a_t)=(s,a) 的次数
    for episode in episodes:
        for i in range(len(episode)):
            s_opt, a_opt, r, s_next = episode[i]
            total_times[i] += 1
            if s == s_opt and a == a_opt:
                occur_times[i] += 1
    for i in reversed(range(timestep_max)):
        if total_times[i]:
            rho += gamma ** i * occur_times[i] / total_times[i]
    return (1 - gamma) * rho


def demo_occupancy() -> None:
    timestep_max = 1000
    episodes_1 = sample(MDP, Pi_1, timestep_max, 1000)
    episodes_2 = sample(MDP, Pi_2, timestep_max, 1000)
    rho_1 = occupancy(episodes_1, "s4", "概率前往", timestep_max, gamma)
    rho_2 = occupancy(episodes_2, "s4", "概率前往", timestep_max, gamma)
    print(rho_1, rho_2)


def main() -> None:
    print("=" * 60)
    print("[1] 用定义计算回报")
    demo_return()

    print("=" * 60)
    print("[2] 贝尔曼方程矩阵形式求 MRP 解析解")
    demo_analytic()

    print("=" * 60)
    print("[3] MDP 策略对应的状态价值解析解")
    demo_analytic_mdp()

    print("=" * 60)
    print("[4] 按策略采样 5 条轨迹")
    demo_sample()

    print("=" * 60)
    print("[5] 蒙特卡洛估计状态价值")
    demo_mc()

    print("=" * 60)
    print("[6] 两个策略在 (s4, 概率前往) 上的占用度量")
    demo_occupancy()


if __name__ == "__main__":
    main()
