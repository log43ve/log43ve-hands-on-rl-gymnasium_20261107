"""第 13 章 DDPG 算法

对应书中示例：在 Pendulum 上用 DDPG（确定性策略梯度 + 目标网络软更新）训练。

运行（需要 torch）：
    python chapter13_ddpg.py
    python chapter13_ddpg.py --episodes 50
"""

from __future__ import annotations

import argparse
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from hrl.plotting import plot_returns
from hrl.rl_utils import ReplayBuffer, get_device, train_off_policy_agent


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # 环境允许的动作最大值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)  # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class DDPG:
    """DDPG 算法。"""

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma,
                 actor_lr, critic_lr, tau, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim,
                               action_bound).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim,
                                      action_bound).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim,
                                       action_dim).to(device)
        # 目标网络初始化为与在线网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声标准差，均值为 0
        self.tau = tau  # 目标网络软更新系数
        self.action_dim = action_dim
        self.device = device

    def take_action(self, state):
        state = torch.as_tensor(np.array([state], dtype=np.float32),
                                device=self.device)
        action = self.actor(state).item()
        # 给动作加噪声以增加探索
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def soft_update(self, net, target_net) -> None:
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict) -> None:
        states = torch.tensor(transition_dict["states"],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict["actions"],
                               dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict["rewards"],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict["next_states"],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict["dones"],
                             dtype=torch.float).view(-1, 1).to(self.device)

        next_q_values = self.target_critic(
            next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)


def main() -> None:
    parser = argparse.ArgumentParser(description="第 13 章 DDPG 算法")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    hidden_dim = 64
    gamma = 0.98
    tau = 0.005
    buffer_size = 10000
    minimal_size = 1000
    batch_size = 64
    sigma = 0.01
    device = get_device(args.cuda)

    env_name = "Pendulum-v1"
    env = gym.make(env_name)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    replay_buffer = ReplayBuffer(buffer_size)
    agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma,
                 actor_lr=3e-4, critic_lr=3e-3, tau=tau, gamma=gamma,
                 device=device)

    return_list = train_off_policy_agent(env, agent, args.episodes,
                                         replay_buffer, minimal_size,
                                         batch_size, seed=args.seed)
    print(f"最后 20 回合平均回报: {np.mean(return_list[-20:]):.3f}")
    path = plot_returns(return_list, f"DDPG on {env_name}", "ch13_ddpg.png")
    print(f"图像已保存: {path}")
    env.close()


if __name__ == "__main__":
    main()
