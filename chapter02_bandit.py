"""第 2 章 多臂老虎机（multi-armed bandit）

对应书中示例：伯努利老虎机、epsilon-贪婪、epsilon 衰减、UCB、汤普森采样。

运行：
    python chapter02_bandit.py                 # 全部实验
    python chapter02_bandit.py --algo ucb      # 只跑 UCB
    python chapter02_bandit.py --steps 10000   # 改变总步数

说明：本章只涉及 numpy 随机数，不依赖 gymnasium。
"""

from __future__ import annotations

import argparse

import numpy as np

from hrl.plotting import plt, save_fig


class BernoulliBandit:
    """伯努利多臂老虎机，输入 K 表示拉杆个数。"""

    def __init__(self, K: int, seed: int | None = None):
        # 沿用书中的全局随机数（numpy legacy API），以便复现书中给出的数值
        if seed is not None:
            np.random.seed(seed)
        # 随机生成 K 个 0~1 的数，作为拉动每根拉杆的获奖概率
        self.probs = np.random.uniform(size=K)
        self.best_idx = int(np.argmax(self.probs))  # 获奖概率最大的拉杆
        self.best_prob = float(self.probs[self.best_idx])  # 最大的获奖概率
        self.K = K

    def step(self, k: int) -> int:
        """选择 k 号拉杆，按概率返回奖励 1（获奖）或 0（未获奖）。"""
        return int(np.random.rand() < self.probs[k])


class Solver:
    """多臂老虎机算法基本框架。"""

    def __init__(self, bandit: BernoulliBandit, seed: int | None = None):
        self.bandit = bandit
        self.counts = np.zeros(bandit.K)  # 每根拉杆的尝试次数
        self.regret = 0.0  # 当前步的累积懊悔
        self.actions = []  # 记录每一步的动作
        self.regrets = []  # 记录每一步的累积懊悔
        if seed is not None:
            np.random.seed(seed)

    def update_regret(self, k: int) -> None:
        """计算累积懊悔并保存，k 为本次动作选择的拉杆编号。"""
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self) -> int:
        raise NotImplementedError

    def run(self, num_steps: int) -> None:
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


class EpsilonGreedy(Solver):
    """epsilon-贪婪算法。"""

    def __init__(self, bandit, epsilon: float = 0.01, init_prob: float = 1.0,
                 seed: int | None = None):
        super().__init__(bandit, seed=seed)
        self.epsilon = epsilon
        # 初始化拉动所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self) -> int:
        if np.random.random() < self.epsilon:
            k = int(np.random.randint(0, self.bandit.K))  # 随机选择一根拉杆
        else:
            k = int(np.argmax(self.estimates))  # 选择期望奖励估值最大的拉杆
        r = self.bandit.step(k)
        # 增量式更新均值：只用到该拉杆的尝试次数
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


class DecayingEpsilonGreedy(Solver):
    """epsilon 值随时间衰减的 epsilon-贪婪算法。"""

    def __init__(self, bandit, init_prob: float = 1.0, seed: int | None = None):
        super().__init__(bandit, seed=seed)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self) -> int:
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:  # epsilon 随时间衰减
            k = int(np.random.randint(0, self.bandit.K))
        else:
            k = int(np.argmax(self.estimates))
        r = self.bandit.step(k)
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


class UCB(Solver):
    """UCB 算法（上置信界）。"""

    def __init__(self, bandit, coef: float, init_prob: float = 1.0,
                 seed: int | None = None):
        super().__init__(bandit, seed=seed)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self) -> int:
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1))
        )  # 计算上置信界
        k = int(np.argmax(ucb))  # 选出上置信界最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


class ThompsonSampling(Solver):
    """汤普森采样算法。"""

    def __init__(self, bandit, seed: int | None = None):
        super().__init__(bandit, seed=seed)
        self._a = np.ones(self.bandit.K)  # 每根拉杆奖励为 1 的次数
        self._b = np.ones(self.bandit.K)  # 每根拉杆奖励为 0 的次数

    def run_one_step(self) -> int:
        samples = np.random.beta(self._a, self._b)  # 按 Beta 分布采样一组奖励
        k = int(np.argmax(samples))
        r = self.bandit.step(k)
        self._a[k] += r
        self._b[k] += 1 - r
        return k


# --------------------------------------------------------------------------- 实验
def plot_results(solvers, solver_names, title: str, filename: str):
    """累积懊悔随步数变化的图像。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for solver, name in zip(solvers, solver_names):
        ax.plot(range(len(solver.regrets)), solver.regrets, label=name)
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Cumulative regrets")
    ax.set_title(title)
    ax.legend()
    path = save_fig(fig, filename)
    print(f"  图像已保存: {path}")


def run_epsilon_greedy(bandit, num_steps: int, epsilon: float = 0.01, seed: int = 1):
    """书中示例：epsilon=0.01 的 epsilon-贪婪，累积懊悔约 25.53。"""
    solver = EpsilonGreedy(bandit, epsilon=epsilon, seed=seed)
    solver.run(num_steps)
    print("epsilon-贪婪算法的累积懊悔为：", solver.regret)
    plot_results([solver], ["EpsilonGreedy"], f"{bandit.K}-armed bandit",
                 "ch02_epsilon_greedy.png")
    return solver


def run_epsilon_comparison(bandit, num_steps: int, seed: int = 0):
    """书中示例：不同 epsilon 取值的对比。"""
    epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
    solvers = []
    for e in epsilons:
        solver = EpsilonGreedy(bandit, epsilon=e, seed=seed)
        solver.run(num_steps)
        solvers.append(solver)
        print(f"  epsilon={e:<7} 累积懊悔 = {solver.regret:.4f}")
    plot_results(solvers, [f"epsilon={e}" for e in epsilons],
                 f"{bandit.K}-armed bandit", "ch02_epsilon_compare.png")
    return solvers


def run_decaying_epsilon(bandit, num_steps: int, seed: int = 1):
    """书中示例：epsilon 随时间衰减，累积懊悔约 10.11。"""
    solver = DecayingEpsilonGreedy(bandit, seed=seed)
    solver.run(num_steps)
    print("epsilon 值衰减的贪婪算法的累积懊悔为：", solver.regret)
    plot_results([solver], ["DecayingEpsilonGreedy"], f"{bandit.K}-armed bandit",
                 "ch02_decaying_epsilon.png")
    return solver


def run_ucb(bandit, num_steps: int, coef: float = 1.0, seed: int = 1):
    """书中示例：UCB，累积懊悔约 70.45。"""
    solver = UCB(bandit, coef, seed=seed)
    solver.run(num_steps)
    print("上置信界算法的累积懊悔为：", solver.regret)
    plot_results([solver], ["UCB"], f"{bandit.K}-armed bandit", "ch02_ucb.png")
    return solver


def run_thompson(bandit, num_steps: int, seed: int = 1):
    """书中示例：汤普森采样，累积懊悔约 57.19。"""
    solver = ThompsonSampling(bandit, seed=seed)
    solver.run(num_steps)
    print("汤普森采样算法的累积懊悔为：", solver.regret)
    plot_results([solver], ["ThompsonSampling"], f"{bandit.K}-armed bandit",
                 "ch02_thompson.png")
    return solver


def main() -> None:
    parser = argparse.ArgumentParser(description="第 2 章 多臂老虎机")
    parser.add_argument("--algo", default="all",
                        choices=["all", "eps", "compare", "decay", "ucb", "thompson"],
                        help="选择要运行的算法")
    parser.add_argument("--steps", type=int, default=5000, help="总交互步数")
    parser.add_argument("--K", type=int, default=10, help="拉杆个数")
    parser.add_argument("--seed", type=int, default=1, help="随机种子")
    args = parser.parse_args()

    bandit = BernoulliBandit(args.K, seed=args.seed)
    print(f"随机生成了一个 {args.K} 臂伯努利老虎机")
    print(f"获奖概率最大的拉杆为 {bandit.best_idx} 号，"
          f"其获奖概率为 {bandit.best_prob:.4f}\n")

    if args.algo in ("all", "eps"):
        print("[1] epsilon-贪婪")
        run_epsilon_greedy(bandit, args.steps, seed=args.seed)
        print()
    if args.algo in ("all", "compare"):
        print("[2] 不同 epsilon 的对比")
        run_epsilon_comparison(bandit, args.steps)
        print()
    if args.algo in ("all", "decay"):
        print("[3] epsilon 衰减")
        run_decaying_epsilon(bandit, args.steps, seed=args.seed)
        print()
    if args.algo in ("all", "ucb"):
        print("[4] UCB")
        run_ucb(bandit, args.steps, seed=args.seed)
        print()
    if args.algo in ("all", "thompson"):
        print("[5] 汤普森采样")
        run_thompson(bandit, args.steps, seed=args.seed)


if __name__ == "__main__":
    main()
