"""第 7 章 DQN 算法

对应书中示例：在 CartPole 上用 DQN 训练，并给出经验回放池、目标网络、
以及用于 Atari 图像的卷积 Q 网络（书中仅展示结构，不训练）。

运行（需要 torch）：
    python chapter07_dqn.py                 # 书中默认 500 回合
    python chapter07_dqn.py --episodes 100  # 快速跑一小段

注意：gymnasium 中 CartPole 只有 v1（v0 已移除），单回合上限 500 步，
因此回报数值与书中 v0（上限 200 步）不能直接比较。
"""

from __future__ import annotations

import argparse
import collections
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from hrl.plotting import plot_returns
from hrl.rl_utils import get_device, train_off_policy_agent


class ReplayBuffer:
    """经验回放池（书中第 7 章引入；后续章节使用 hrl.rl_utils.ReplayBuffer）。"""

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)  # 队列，先进先出

    def add(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self) -> int:
        return len(self.buffer)


class Qnet(torch.nn.Module):
    """只有一层隐藏层的 Q 网络。"""

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用 ReLU
        return self.fc2(x)


class DQN:
    """DQN 算法。"""

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.action_dim))
        state = torch.as_tensor(np.array([state], dtype=np.float32), device=self.device)
        return int(self.q_net(state).argmax().item())

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

        q_values = self.q_net(states).gather(1, actions)  # Q 值
        # 下一个状态的最大 Q 值（用目标网络）
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


class ConvolutionalQnet(torch.nn.Module):
    """加入卷积层的 Q 网络（书中用于 Atari 图像输入，这里仅给出结构）。"""

    def __init__(self, action_dim: int, in_channels: int = 4):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = torch.nn.Linear(7 * 7 * 64, 512)
        self.head = torch.nn.Linear(512, action_dim)

    def forward(self, x):
        x = x / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x))
        return self.head(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="第 7 章 DQN 算法")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true", help="有 GPU 时使用 GPU")
    args = parser.parse_args()

    hidden_dim = 128
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = get_device(args.cuda)

    env_name = "CartPole-v1"
    env = gym.make(env_name)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = int(env.action_space.n)
    replay_buffer = ReplayBuffer(buffer_size)
    agent = DQN(state_dim, hidden_dim, action_dim, args.lr, args.gamma,
                args.epsilon, target_update, device)

    return_list = train_off_policy_agent(env, agent, args.episodes,
                                         replay_buffer, minimal_size,
                                         batch_size, seed=args.seed)

    print(f"最后 20 回合平均回报: {np.mean(return_list[-20:]):.3f}")
    path = plot_returns(return_list, f"DQN on {env_name}", "ch07_dqn.png")
    print(f"图像已保存: {path}")
    env.close()


if __name__ == "__main__":
    main()
