"""第 4 章 动态规划算法（dynamic programming）

对应书中示例：
  * 策略迭代（policy iteration）在悬崖漫步上的结果
  * 价值迭代（value iteration）在悬崖漫步上的结果
  * 在 gymnasium 的 FrozenLake-v1 上重复以上两个算法

环境通过 gymnasium 创建，直接使用 ``env.unwrapped.P`` 中的转移模型
（``P[s][a] = [(prob, next_state, reward, terminated)]``），
因此同一套 DP 代码既能跑书中的悬崖漫步，也能跑 gymnasium 的冰湖。

运行：
    python chapter04_dp.py                 # 两个环境 + 两个算法
    python chapter04_dp.py --env frozen    # 只跑冰湖
    python chapter04_dp.py --algo vi       # 只跑价值迭代
"""

from __future__ import annotations

import argparse
import copy

import numpy as np

from hrl.envs import grid_shape, make_env


def _unwrap(env):
    """去掉 TimeLimit 等包装层，拿到带 P / nrow / ncol 的原始环境。"""
    return env.unwrapped if hasattr(env, "unwrapped") else env


class PolicyIteration:
    """策略迭代算法：策略评估 + 策略提升交替进行。"""

    def __init__(self, env, theta: float = 0.001, gamma: float = 0.9):
        self.env = _unwrap(env)
        self.nS = int(self.env.observation_space.n)
        self.nA = int(self.env.action_space.n)
        self.v = np.zeros(self.nS)  # 状态价值初始化为 0
        self.pi = np.full((self.nS, self.nA), 1.0 / self.nA)  # 均匀随机策略
        self.theta = theta  # 策略评估收敛阈值
        self.gamma = gamma  # 折扣因子

    def _qsa(self, s: int, a: int) -> float:
        """计算 Q(s,a) = Σ p (r + γ V(s'))。"""
        qsa = 0.0
        for prob, next_state, reward, terminated in self.env.P[s][a]:
            qsa += prob * (reward + self.gamma * self.v[next_state] * (1 - terminated))
        return qsa

    def policy_evaluation(self, verbose: bool = True) -> None:
        cnt = 1
        while True:
            max_diff = 0.0
            new_v = np.zeros(self.nS)
            for s in range(self.nS):
                new_v[s] = sum(
                    self.pi[s][a] * self._qsa(s, a) for a in range(self.nA)
                )
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:  # 满足收敛条件，退出评估迭代
                break
            cnt += 1
        if verbose:
            print("策略评估进行%d轮后完成" % cnt)

    def policy_improvement(self, verbose: bool = True) -> None:
        for s in range(self.nS):
            qsa_list = [self._qsa(s, a) for a in range(self.nA)]
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)  # 有几个动作取得了最大的 Q 值
            # 让这些动作均分概率（各占 1/cntq）
            self.pi[s] = np.array([1 / cntq if q == maxq else 0.0 for q in qsa_list])
        if verbose:
            print("策略提升完成")

    def policy_iteration(self) -> None:
        while True:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)
            self.policy_improvement()
            if np.array_equal(old_pi, self.pi):
                break


class ValueIteration:
    """价值迭代算法：用 max 替代期望，只维护状态价值。"""

    def __init__(self, env, theta: float = 0.001, gamma: float = 0.9):
        self.env = _unwrap(env)
        self.nS = int(self.env.observation_space.n)
        self.nA = int(self.env.action_space.n)
        self.v = np.zeros(self.nS)
        self.theta = theta
        self.gamma = gamma
        self.pi = [None] * self.nS  # 价值迭代结束后导出的策略

    def _qsa(self, s: int, a: int) -> float:
        qsa = 0.0
        for prob, next_state, reward, terminated in self.env.P[s][a]:
            qsa += prob * (reward + self.gamma * self.v[next_state] * (1 - terminated))
        return qsa

    def value_iteration(self) -> None:
        cnt = 0
        while True:
            max_diff = 0.0
            new_v = np.zeros(self.nS)
            for s in range(self.nS):
                qsa_list = [self._qsa(s, a) for a in range(self.nA)]
                new_v[s] = max(qsa_list)  # 与策略迭代的主要区别
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:  # 满足收敛条件，退出迭代
                break
            cnt += 1
        print("价值迭代一共进行%d轮" % cnt)
        self.get_policy()

    def get_policy(self) -> None:
        """根据价值函数导出贪婪策略。"""
        for s in range(self.nS):
            qsa_list = [self._qsa(s, a) for a in range(self.nA)]
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)
            self.pi[s] = [1 / cntq if q == maxq else 0.0 for q in qsa_list]


def print_agent(agent, action_meaning, disaster=(), end=()):
    """打印状态价值表格与策略表格。"""
    nrow, ncol = grid_shape(agent.env)
    print("状态价值：")
    for i in range(nrow):
        for j in range(ncol):
            print("%6.3f" % agent.v[i * ncol + j], end=" ")
        print()

    print("策略：")
    for i in range(nrow):
        for j in range(ncol):
            s = i * ncol + j
            if s in disaster:  # 悬崖 / 冰洞
                print("****", end=" ")
            elif s in end:  # 目标状态
                print("EEEE", end=" ")
            else:
                a = agent.pi[s]
                pi_str = "".join(
                    action_meaning[k] if a[k] > 0 else "o" for k in range(len(action_meaning))
                )
                print(pi_str, end=" ")
        print()


def run_cliff(gamma: float = 0.9, theta: float = 0.001) -> None:
    """悬崖漫步：书中 4.2 / 4.4 节的实验。"""
    env = make_env("book", render_mode=None)
    action_meaning = ["^", "v", "<", ">"]
    disaster, end = list(range(37, 47)), [47]

    print("-" * 60)
    print("[悬崖漫步] 策略迭代")
    agent = PolicyIteration(env, theta=theta, gamma=gamma)
    agent.policy_iteration()
    print_agent(agent, action_meaning, disaster, end)

    print("-" * 60)
    print("[悬崖漫步] 价值迭代")
    agent = ValueIteration(env, theta=theta, gamma=gamma)
    agent.value_iteration()
    print_agent(agent, action_meaning, disaster, end)
    env.close()


def run_frozen(gamma: float = 0.9, theta: float = 1e-5) -> None:
    """冰湖（gymnasium FrozenLake-v1）：书中 4.5 节的实验。"""
    env = make_env("frozen", render_mode="ansi")
    env.reset(seed=0)
    print(env.render())

    unwrapped = env.unwrapped
    holes, ends = set(), set()
    for s in unwrapped.P:
        for a in unwrapped.P[s]:
            for trans in unwrapped.P[s][a]:
                if trans[2] == 1.0:  # 获得奖励 1，代表是目标
                    ends.add(trans[1])
                if trans[3] is True:  # 终止状态
                    holes.add(trans[1])
    holes = holes - ends
    print("冰洞的索引:", holes)
    print("目标的索引:", ends)
    for a in range(env.action_space.n):  # 查看目标左边一格的状态转移信息
        print(unwrapped.P[14][a])

    action_meaning = ["<", "v", ">", "^"]  # gymnasium 冰湖规定的动作顺序
    print("-" * 60)
    print("[冰湖] 策略迭代")
    agent = PolicyIteration(env, theta=theta, gamma=gamma)
    agent.policy_iteration()
    print_agent(agent, action_meaning, sorted(holes), sorted(ends))

    print("-" * 60)
    print("[冰湖] 价值迭代")
    agent = ValueIteration(env, theta=theta, gamma=gamma)
    agent.value_iteration()
    print_agent(agent, action_meaning, sorted(holes), sorted(ends))
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="第 4 章 动态规划算法")
    parser.add_argument("--env", default="both", choices=["cliff", "frozen", "both"])
    parser.add_argument("--algo", default="both", choices=["pi", "vi", "both"])
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--theta", type=float, default=None,
                        help="收敛阈值，默认悬崖漫步 1e-3、冰湖 1e-5")
    args = parser.parse_args()

    if args.env in ("cliff", "both"):
        run_cliff(args.gamma, args.theta if args.theta is not None else 0.001)
    if args.env in ("frozen", "both"):
        run_frozen(args.gamma, args.theta if args.theta is not None else 1e-5)


if __name__ == "__main__":
    main()
