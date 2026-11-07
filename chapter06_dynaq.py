"""第 6 章 Dyna-Q 算法

对应书中示例：在悬崖漫步上比较 Q-planning 步数为 0 / 2 / 20 时的学习曲线。

运行：
    python chapter06_dynaq.py
    python chapter06_dynaq.py --planning 0 5 50
    python chapter06_dynaq.py --env gym     # 换成 gymnasium 内置 CliffWalking-v1
"""

from __future__ import annotations

import argparse
import random

import numpy as np
from tqdm import tqdm

from hrl.envs import make_env
from hrl.plotting import plt, save_fig


class DynaQ:
    """Dyna-Q 算法：Q-learning + 环境模型 + Q-planning。"""

    def __init__(self, n_states, n_actions, epsilon=0.01, alpha=0.1,
                 gamma=0.9, n_planning=0, seed=0):
        self.Q_table = np.zeros([n_states, n_actions])
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning = n_planning  # 执行 Q-planning 的次数
        self.model = dict()  # 环境模型：(s, a) -> (r, s')
        self.rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)

    def take_action(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.Q_table[state]))

    def q_learning(self, s0, a0, r, s1, done=False):
        td_error = r + self.gamma * self.Q_table[s1].max() * (1 - done) - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

    def update(self, s0, a0, r, s1, done=False):
        self.q_learning(s0, a0, r, s1, done)
        self.model[(s0, a0)] = (r, s1, done)  # 将数据添加到模型中
        for _ in range(self.n_planning):  # Q-planning 循环
            # 随机选择曾经遇到过的状态动作对
            (s, a), (r, s_, done_) = self.py_rng.choice(list(self.model.items()))
            self.q_learning(s, a, r, s_, done_)


def train_dynaq(env, n_planning: int, num_episodes: int, seed: int = 0,
                epsilon: float = 0.01, alpha: float = 0.1, gamma: float = 0.9):
    """训练一个 Dyna-Q 智能体，返回每条序列的回报列表。"""
    n_states = int(env.observation_space.n)
    n_actions = int(env.action_space.n)
    agent = DynaQ(n_states, n_actions, epsilon, alpha, gamma, n_planning, seed)

    return_list = []
    env.reset(seed=seed)
    pbar = tqdm(total=num_episodes, ncols=80, disable=None,
                desc=f"planning={n_planning:<3}")
    for i_episode in range(num_episodes):
        episode_return = 0.0
        state, info = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward  # 这里的回报不做折扣衰减
            agent.update(state, action, reward, next_state, done)
            state = next_state
        return_list.append(episode_return)
        if (i_episode + 1) % 10 == 0:
            pbar.set_postfix({"return": "%.3f" % np.mean(return_list[-10:])})
        pbar.update(1)
    pbar.close()
    return return_list, agent


def main() -> None:
    parser = argparse.ArgumentParser(description="第 6 章 Dyna-Q 算法")
    parser.add_argument("--env", default="book", choices=["book", "gym"])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--planning", type=int, nargs="+", default=[0, 2, 20],
                        help="Q-planning 步数列表")
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = make_env(args.env, render_mode=None)
    env_name = "Cliff Walking"
    fig, ax = plt.subplots(figsize=(8, 5))

    for n_planning in args.planning:
        print("=" * 60)
        print("Q-planning 步数为：%d" % n_planning)
        return_list, agent = train_dynaq(env, n_planning, args.episodes,
                                         args.seed, args.epsilon, args.alpha,
                                         args.gamma)
        print(f"最后 20 回合平均回报: {np.mean(return_list[-20:]):.3f}")
        ax.plot(range(len(return_list)), return_list,
                label=f"{n_planning} planning steps")

    ax.legend()
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Returns")
    ax.set_title(f"Dyna-Q on {env_name}")
    path = save_fig(fig, "ch06_dynaq.png")
    print(f"\n图像已保存: {path}")
    env.close()


if __name__ == "__main__":
    main()
