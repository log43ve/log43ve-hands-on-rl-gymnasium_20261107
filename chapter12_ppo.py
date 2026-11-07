"""第 12 章 PPO 算法

对应书中示例：
  * 离散动作（CartPole）上的 PPO（截断目标）
  * 连续动作（Pendulum）上的 PPO（截断目标）

运行（需要 torch）：
    python chapter12_ppo.py --env cartpole --episodes 200
    python chapter12_ppo.py --env pendulum --episodes 100   # 书中默认 2000
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from hrl.plotting import plot_returns
from hrl.rl_utils import compute_advantage, get_device, train_on_policy_agent


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    """PPO，采用截断（clip）方式。"""

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练多少轮
        self.eps = eps  # 截断范围的参数
        self.device = device

    def take_action(self, state) -> int:
        state = torch.as_tensor(np.array([state], dtype=np.float32),
                                device=self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        return int(action_dist.sample().item())

    def update(self, transition_dict) -> None:
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
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                      td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO 损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std


class PPOContinuous:
    """处理连续动作的 PPO。"""

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim,
                                         action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.as_tensor(np.array([state], dtype=np.float32),
                                device=self.device)
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return [action.item()]

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
        rewards = (rewards + 8.0) / 8.0  # 与 TRPO 一样对奖励做缩放
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                      td_delta.cpu()).to(self.device)
        mu, std = self.actor(states)
        old_action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = old_action_dists.log_prob(actions)

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


def main() -> None:
    parser = argparse.ArgumentParser(description="第 12 章 PPO 算法")
    parser.add_argument("--env", default="cartpole",
                        choices=["cartpole", "pendulum"])
    parser.add_argument("--episodes", type=int, default=None,
                        help="默认 CartPole 500、Pendulum 2000（书中设置）")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    hidden_dim = 128
    lmbda = 0.95
    eps = 0.2
    device = get_device(args.cuda)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.env == "cartpole":
        env_name = "CartPole-v1"
        env = gym.make(env_name)
        episodes = args.episodes if args.episodes is not None else 500
        agent = PPO(env.observation_space.shape[0], hidden_dim,
                    int(env.action_space.n), actor_lr=1e-3, critic_lr=1e-2,
                    lmbda=lmbda, epochs=args.epochs, eps=eps, gamma=0.98,
                    device=device)
        tag = "ch12_ppo_cartpole.png"
    else:
        env_name = "Pendulum-v1"
        env = gym.make(env_name)
        episodes = args.episodes if args.episodes is not None else 2000
        agent = PPOContinuous(env.observation_space.shape[0], hidden_dim,
                              env.action_space.shape[0], actor_lr=1e-4,
                              critic_lr=5e-3, lmbda=0.9,
                              epochs=args.epochs, eps=eps, gamma=0.9,
                              device=device)
        tag = "ch12_ppo_pendulum.png"

    env.reset(seed=args.seed)
    return_list = train_on_policy_agent(env, agent, episodes, seed=args.seed)
    print(f"最后 20 回合平均回报: {np.mean(return_list[-20:]):.3f}")
    path = plot_returns(return_list, f"PPO on {env_name}", tag)
    print(f"图像已保存: {path}")
    env.close()


if __name__ == "__main__":
    main()
