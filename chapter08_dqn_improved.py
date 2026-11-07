"""第 8 章 DQN 改进算法

对应书中示例，在 Pendulum（连续动作离散化）上比较：
  * 原始 DQN（观察 Q 值高估）
  * Double DQN（缓解高估）
  * Dueling DQN（价值与优势分解）

运行（需要 torch）：
    python chapter08_dqn_improved.py
    python chapter08_dqn_improved.py --variants vanilla double --episodes 60
"""

from __future__ import annotations

import argparse
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from hrl.plotting import plt, save_fig
from hrl.rl_utils import ReplayBuffer, get_device, moving_average


class Qnet(torch.nn.Module):
    """只有一层隐藏层的 Q 网络。"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class VAnet(torch.nn.Module):
    """Dueling DQN：共享特征层，分别输出 V 与 A。"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        Q = V + A - A.mean(1).view(-1, 1)  # Q = V + A - mean(A)
        return Q


class DQN:
    """DQN / Double DQN / Dueling DQN。"""

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device, dqn_type="VanillaDQN"):
        self.action_dim = action_dim
        if dqn_type == "DuelingDQN":
            self.q_net = VAnet(state_dim, hidden_dim, action_dim).to(device)
            self.target_q_net = VAnet(state_dim, hidden_dim, action_dim).to(device)
        else:
            self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
            self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.dqn_type = dqn_type
        self.device = device

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.action_dim))
        state = torch.as_tensor(np.array([state], dtype=np.float32), device=self.device)
        return int(self.q_net(state).argmax().item())

    def max_q_value(self, state) -> float:
        state = torch.as_tensor(np.array([state], dtype=np.float32), device=self.device)
        return float(self.q_net(state).max().item())

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict["rewards"],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict["next_states"],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict["dones"],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        if self.dqn_type == "DoubleDQN":  # 与 DQN 的唯一区别
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else:
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


def dis_to_con(discrete_action: int, env, action_dim: int) -> float:
    """把离散动作编号映射回连续动作取值。"""
    action_lowbound = env.action_space.low[0]
    action_upbound = env.action_space.high[0]
    return float(action_lowbound + (discrete_action / (action_dim - 1)) *
                 (action_upbound - action_lowbound))


def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size,
              batch_size, seed=0):
    """训练并记录每回合回报与平滑后的最大 Q 值。"""
    from tqdm import tqdm

    return_list = []
    max_q_value_list = []
    max_q_value = 0.0
    state, info = env.reset(seed=seed)
    pbar = tqdm(total=num_episodes, ncols=80, disable=None, desc="train")
    for i_episode in range(num_episodes):
        if i_episode > 0:
            state, info = env.reset()
        episode_return = 0.0
        done = False
        while not done:
            action = agent.take_action(state)
            max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995
            max_q_value_list.append(max_q_value)
            action_continuous = dis_to_con(action, env, agent.action_dim)
            next_state, reward, terminated, truncated, info = env.step(
                [action_continuous])
            done = terminated or truncated
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                agent.update({"states": b_s, "actions": b_a, "rewards": b_r,
                              "next_states": b_ns, "dones": b_d})
        return_list.append(episode_return)
        if (i_episode + 1) % 10 == 0:
            pbar.set_postfix({"return": "%.3f" % np.mean(return_list[-10:])})
        pbar.update(1)
    pbar.close()
    return return_list, max_q_value_list


def plot_q_value(max_q_value_list, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(max_q_value_list)), max_q_value_list)
    ax.axhline(0, c="orange", ls="--")
    ax.axhline(10, c="red", ls="--")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Q value")
    ax.set_title(title)
    return save_fig(fig, filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="第 8 章 DQN 改进算法")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variants", nargs="+",
                        default=["vanilla", "double", "dueling"],
                        choices=["vanilla", "double", "dueling"])
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    lr = 1e-2
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 50
    buffer_size = 5000
    minimal_size = 1000
    batch_size = 64
    action_dim = 11  # 将连续动作分成 11 个离散动作
    device = get_device(args.cuda)

    env_name = "Pendulum-v1"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]

    specs = {
        "vanilla": ("VanillaDQN", "DQN"),
        "double": ("DoubleDQN", "Double DQN"),
        "dueling": ("DuelingDQN", "Dueling DQN"),
    }

    for key in args.variants:
        dqn_type, title = specs[key]
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        replay_buffer = ReplayBuffer(buffer_size)
        agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                    target_update, device, dqn_type)
        print(f"\n[{title}] 训练 {args.episodes} 个回合 ...")
        return_list, max_q_value_list = train_DQN(
            agent, env, args.episodes, replay_buffer, minimal_size,
            batch_size, seed=args.seed)
        print(f"{title} 最后 20 回合平均回报: {np.mean(return_list[-20:]):.3f}")
        p1 = plot_returns_curve(return_list, f"{title} on {env_name}",
                                f"ch08_{key}_return.png")
        p2 = plot_q_value(max_q_value_list, f"{title} on {env_name}",
                          f"ch08_{key}_qvalue.png")
        print(f"图像已保存: {p1}, {p2}")
    env.close()


def plot_returns_curve(return_list, title, filename):
    """平滑后的回报曲线（书中 ch8 用的窗口是 5）。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    mv = moving_average(return_list, 5)
    ax.plot(range(len(mv)), mv)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Returns")
    ax.set_title(title)
    return save_fig(fig, filename)


if __name__ == "__main__":
    main()
