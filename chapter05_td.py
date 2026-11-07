"""第 5 章 时序差分算法（temporal-difference learning）

对应书中示例：
  * Sarsa 在悬崖漫步上的学习曲线与最终策略
  * n 步 Sarsa（默认 n=5）
  * Q-learning

环境使用 gymnasium 接口：
    obs, info = env.reset(seed=...)
    obs, reward, terminated, truncated, info = env.step(action)

运行：
    python chapter05_td.py                 # 三个算法依次运行
    python chapter05_td.py --algo qlearning
    python chapter05_td.py --env gym       # 换成 gymnasium 内置 CliffWalking-v1
"""

from __future__ import annotations

import argparse

import numpy as np
from tqdm import tqdm

from hrl.envs import grid_shape, make_env
from hrl.plotting import moving_average, plt, save_fig

ACTION_MEANING = ["^", "v", "<", ">"]  # 书中动作顺序：上、下、左、右


# --------------------------------------------------------------------------- 算法
class Sarsa:
    """Sarsa 算法（同策略 TD 控制）。"""

    def __init__(self, n_states, n_actions, epsilon=0.1, alpha=0.1,
                 gamma=0.9, seed=0):
        self.Q_table = np.zeros([n_states, n_actions])
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def take_action(self, state: int) -> int:
        """epsilon-贪婪策略选择动作。"""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.Q_table[state]))

    def best_action(self, state: int):
        """用于打印策略：Q 值最大的动作记 1，可能有多个。"""
        Q_max = np.max(self.Q_table[state])
        return [1 if q == Q_max else 0 for q in self.Q_table[state]]

    def update(self, s0, a0, r, s1, a1, done=False):
        td_error = r + self.gamma * self.Q_table[s1, a1] * (1 - done) - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error


class nstep_Sarsa(Sarsa):
    """n 步 Sarsa 算法。"""

    def __init__(self, n, n_states, n_actions, epsilon=0.1, alpha=0.1,
                 gamma=0.9, seed=0):
        super().__init__(n_states, n_actions, epsilon, alpha, gamma, seed)
        self.n = n  # 采用 n 步 Sarsa
        self.state_list = []  # 保存之前的状态
        self.action_list = []  # 保存之前的动作
        self.reward_list = []  # 保存之前的奖励

    def update(self, s0, a0, r, s1, a1, done=False):
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)
        if len(self.state_list) == self.n:  # 保存的数据足够做一次 n 步更新
            G = self.Q_table[s1, a1] * (1 - done)  # Q(s_{t+n}, a_{t+n})
            for i in reversed(range(self.n)):
                G = self.gamma * G + self.reward_list[i]
                # 若到达终止状态，最后几步长度不够 n 步也进行更新
                if done and i > 0:
                    s = self.state_list[i]
                    a = self.action_list[i]
                    self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
            s = self.state_list.pop(0)  # 需要更新的状态动作出队
            a = self.action_list.pop(0)
            self.reward_list.pop(0)
            self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
        if done:  # 一条序列结束，清空缓存
            self.state_list = []
            self.action_list = []
            self.reward_list = []


class QLearning:
    """Q-learning 算法（异策略 TD 控制）。"""

    def __init__(self, n_states, n_actions, epsilon=0.1, alpha=0.1,
                 gamma=0.9, seed=0):
        self.Q_table = np.zeros([n_states, n_actions])
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def take_action(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.Q_table[state]))

    def best_action(self, state: int):
        Q_max = np.max(self.Q_table[state])
        return [1 if q == Q_max else 0 for q in self.Q_table[state]]

    def update(self, s0, a0, r, s1, done=False):
        td_error = r + self.gamma * self.Q_table[s1].max() * (1 - done) - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error


# ---------------------------------------------------------------------- 训练循环
def _train(env, agent, num_episodes: int, mode: str, seed: int = 0):
    """通用回合训练循环，返回每条序列的回报列表。"""
    return_list = []
    env.reset(seed=seed)  # 只播种一次，之后每个回合 reset 沿用
    pbar = tqdm(total=num_episodes, desc=f"{mode:<10}", ncols=80, disable=None)
    for i_episode in range(num_episodes):
        episode_return = 0.0
        state, info = env.reset()
        action = agent.take_action(state)
        done = False
        while not done:
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_action = agent.take_action(next_state)
            episode_return += reward  # 这里的回报不做折扣衰减
            if mode == "qlearning":
                agent.update(state, action, reward, next_state, done)
            else:
                agent.update(state, action, reward, next_state, next_action, done)
            state, action = next_state, next_action
        return_list.append(episode_return)
        if (i_episode + 1) % 10 == 0:
            pbar.set_postfix({"return": "%.3f" % np.mean(return_list[-10:])})
        pbar.update(1)
    pbar.close()
    return return_list


def print_agent(agent, env, action_meaning, disaster=(), end=()):
    """以表格形式打印最终策略。"""
    nrow, ncol = grid_shape(env)
    for i in range(nrow):
        for j in range(ncol):
            s = i * ncol + j
            if s in disaster:
                print("****", end=" ")
            elif s in end:
                print("EEEE", end=" ")
            else:
                a = agent.best_action(s)
                pi_str = "".join(
                    action_meaning[k] if a[k] > 0 else "o" for k in range(len(action_meaning))
                )
                print(pi_str, end=" ")
        print()


def plot_returns(return_list, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(return_list)), return_list, alpha=0.4, label="return")
    smooth = moving_average(return_list, 20)
    ax.plot(range(len(return_list) - len(smooth), len(return_list)), smooth,
            linewidth=2, label="moving average (20)")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Returns")
    ax.set_title(title)
    ax.legend()
    path = save_fig(fig, filename)
    print(f"  图像已保存: {path}")


# -------------------------------------------------------------------------- 入口
def main() -> None:
    parser = argparse.ArgumentParser(description="第 5 章 时序差分算法")
    parser.add_argument("--algo", default="all",
                        choices=["sarsa", "nstep", "qlearning", "all"])
    parser.add_argument("--env", default="book", choices=["book", "gym"])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--n", type=int, default=5, help="n 步 Sarsa 的 n")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = make_env(args.env, render_mode=None)
    unwrapped = env.unwrapped
    n_states = int(env.observation_space.n)
    n_actions = int(env.action_space.n)
    env_name = "Cliff Walking" if "Cliff" in type(unwrapped).__name__ else type(unwrapped).__name__
    print(f"环境: {type(unwrapped).__name__}  nS={n_states}  nA={n_actions}")
    print(f"超参数: epsilon={args.epsilon} alpha={args.alpha} "
          f"gamma={args.gamma} episodes={args.episodes}\n")

    nrow, ncol = grid_shape(env)
    # 最后一行的第 1~(ncol-1) 格是悬崖，最后一格是终点
    disaster = list(range((nrow - 1) * ncol + 1, nrow * ncol - 1))
    end = [nrow * ncol - 1]

    def build(mode):
        if mode == "sarsa":
            return Sarsa(n_states, n_actions, args.epsilon, args.alpha,
                         args.gamma, args.seed)
        if mode == "nstep":
            return nstep_Sarsa(args.n, n_states, n_actions, args.epsilon,
                               args.alpha, args.gamma, args.seed)
        return QLearning(n_states, n_actions, args.epsilon, args.alpha,
                         args.gamma, args.seed)

    modes = ["sarsa", "nstep", "qlearning"] if args.algo == "all" else [args.algo]
    titles = {
        "sarsa": ("Sarsa", "ch05_sarsa.png"),
        "nstep": (f"{args.n}-step Sarsa", f"ch05_{args.n}step_sarsa.png"),
        "qlearning": ("Q-learning", "ch05_qlearning.png"),
    }

    for mode in modes:
        title, filename = titles[mode]
        print("=" * 60)
        print(f"[{title}]")
        agent = build(mode)
        return_list = _train(env, agent, args.episodes, mode, seed=args.seed)
        print(f"{title} 后 50 回合平均回报: {np.mean(return_list[-50:]):.3f}")
        plot_returns(return_list, f"{title} on {env_name}", filename)
        print(f"{title} 最终收敛得到的策略为：")
        print_agent(agent, env, ACTION_MEANING, disaster, end)
        env.reset(seed=args.seed)

    env.close()


if __name__ == "__main__":
    main()
