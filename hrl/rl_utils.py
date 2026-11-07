"""书中 ``rl_utils.py`` 的 gymnasium 版本。

《动手学强化学习》第 7 章之后的示例都依赖作者提供的 ``rl_utils`` 工具模块
（见 https://github.com/boyu-ai/Hands-on-RL ）。原版按旧版 gym 的三元组接口
（``next_state, reward, done, _ = env.step(a)``）编写，这里改写为 gymnasium 的五元组接口，
并保留原有函数名与签名，方便和书中代码对照：

  * ``ReplayBuffer``              —— 经验回放池
  * ``moving_average``            —— 滑动平均（书中曲线平滑用）
  * ``train_on_policy_agent``     —— 同策略算法（REINFORCE / Actor-Critic / TRPO / PPO / GAIL）训练循环
  * ``train_off_policy_agent``    —— 异策略算法（DQN / DDPG / SAC / CQL）训练循环
  * ``compute_advantage``         —— GAE 优势估计
"""

from __future__ import annotations

import collections
import os
import random

import numpy as np
import torch
from tqdm import tqdm


class ReplayBuffer:
    """经验回放池。"""

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)  # 先进先出队列

    def add(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return (np.array(state), action, reward, np.array(next_state), done)

    def size(self) -> int:
        return len(self.buffer)


def moving_average(a, window_size: int):
    """与书中实现一致的滑动平均（两端用缩小的窗口对齐）。"""
    a = np.asarray(a, dtype=float)
    if len(a) <= window_size:
        return a
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (
        cumulative_sum[window_size:] - cumulative_sum[:-window_size]
    ) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[: window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def _setup_threads() -> None:
    """这些示例的网络很小，CPU 上开多线程反而更慢（线程争用），默认单线程。

    可用环境变量 HRL_TORCH_THREADS 覆盖。
    """
    torch.set_num_threads(int(os.environ.get("HRL_TORCH_THREADS", "1")))


def _reset(env, seed):
    obs, info = env.reset(seed=seed)
    return obs


def train_on_policy_agent(env, agent, num_episodes: int, seed: int = 0):
    """同策略算法的通用训练循环，返回每条序列的回报。"""
    _setup_threads()
    return_list = []
    state = _reset(env, seed)
    pbar = tqdm(total=num_episodes, ncols=80, disable=None, desc="train")
    for i_episode in range(num_episodes):
        if i_episode > 0:
            state, info = env.reset()
        episode_return = 0.0
        transition_dict = {
            "states": [], "actions": [], "next_states": [],
            "rewards": [], "dones": [],
        }
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            transition_dict["states"].append(state)
            transition_dict["actions"].append(action)
            transition_dict["next_states"].append(next_state)
            transition_dict["rewards"].append(reward)
            transition_dict["dones"].append(done)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
        agent.update(transition_dict)
        if (i_episode + 1) % 10 == 0:
            pbar.set_postfix({"return": "%.3f" % np.mean(return_list[-10:])})
        pbar.update(1)
    pbar.close()
    return return_list


def train_off_policy_agent(env, agent, num_episodes: int, replay_buffer,
                           minimal_size: int, batch_size: int, seed: int = 0):
    """异策略算法的通用训练循环，返回每条序列的回报。"""
    _setup_threads()
    return_list = []
    state = _reset(env, seed)
    pbar = tqdm(total=num_episodes, ncols=80, disable=None, desc="train")
    for i_episode in range(num_episodes):
        if i_episode > 0:
            state, info = env.reset()
        episode_return = 0.0
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {
                    "states": b_s, "actions": b_a, "next_states": b_ns,
                    "rewards": b_r, "dones": b_d,
                }
                agent.update(transition_dict)
        return_list.append(episode_return)
        if (i_episode + 1) % 10 == 0:
            pbar.set_postfix({"return": "%.3f" % np.mean(return_list[-10:])})
        pbar.update(1)
    pbar.close()
    return return_list


def compute_advantage(gamma: float, lmbda: float, td_delta):
    """广义优势估计（GAE）。"""
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


def get_device(cuda: bool = False) -> torch.device:
    """默认使用 CPU（这些示例的网络很小，CPU 足够快且不依赖显卡驱动）。"""
    _setup_threads()
    if cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
