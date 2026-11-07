"""第 14 章 SAC 算法

对应书中示例：
  * 连续动作（Pendulum）上的 SAC：双 Q 网络 + 最大熵 + 自动调节温度 alpha
  * 离散动作（CartPole）上的 SAC：按动作概率直接计算熵的期望

运行（需要 torch）：
    python chapter14_sac.py --env pendulum --episodes 40
    python chapter14_sac.py --env cartpole --episodes 100
"""

from __future__ import annotations

import argparse
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from hrl.plotting import plot_returns
from hrl.rl_utils import ReplayBuffer, get_device, train_off_policy_agent


class PolicyNetContinuous(torch.nn.Module):
    """连续动作策略：tanh 变换后的高斯策略，并修正对数概率。"""

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # 重参数化采样
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # tanh 变换后的对数概率密度修正
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound
        return action, log_prob


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class SACContinuous:
    """处理连续动作的 SAC 算法。"""

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim,
                                         action_bound).to(device)
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)
        self.target_critic_1 = QValueNetContinuous(
            state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = QValueNetContinuous(
            state_dim, hidden_dim, action_dim).to(device)
        # 目标 Q 网络初始参数与 Q 网络一致
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # 用 alpha 的 log 值，训练更稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state):
        state = torch.as_tensor(np.array([state], dtype=np.float32),
                                device=self.device)
        action = self.actor(state)[0]
        return [action.item()]

    def calc_target(self, rewards, next_states, dones):
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        return rewards + self.gamma * next_value * (1 - dones)

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
        # 与前面章节一样，对倒立摆的奖励做缩放
        rewards = (rewards + 8.0) / 8.0

        # 更新两个 Q 网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 自动调节温度 alpha
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SAC:
    """处理离散动作的 SAC 算法。"""

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 alpha_lr, target_entropy, tau, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = QValueNet(state_dim, hidden_dim,
                                         action_dim).to(device)
        self.target_critic_2 = QValueNet(state_dim, hidden_dim,
                                         action_dim).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state) -> int:
        state = torch.as_tensor(np.array([state], dtype=np.float32),
                                device=self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        return int(action_dist.sample().item())

    def calc_target(self, rewards, next_states, dones):
        """用策略输出的动作概率直接求熵与 Q 的期望。"""
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                               dim=1, keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        return rewards + self.gamma * next_value * (1 - dones)

    def soft_update(self, net, target_net) -> None:
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict) -> None:
        states = torch.tensor(transition_dict["states"],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(
            self.device)  # 离散动作，不要转成 float
        rewards = torch.tensor(transition_dict["rewards"],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict["next_states"],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict["dones"],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 更新两个 Q 网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1, keepdim=True)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 自动调节温度 alpha
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


def main() -> None:
    parser = argparse.ArgumentParser(description="第 14 章 SAC 算法")
    parser.add_argument("--env", default="pendulum",
                        choices=["pendulum", "cartpole"])
    parser.add_argument("--episodes", type=int, default=None,
                        help="默认 Pendulum 100、CartPole 200（书中设置）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    hidden_dim = 128
    tau = 0.005
    batch_size = 64
    device = get_device(args.cuda)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.env == "pendulum":
        env_name = "Pendulum-v1"
        env = gym.make(env_name)
        episodes = args.episodes if args.episodes is not None else 100
        action_bound = env.action_space.high[0]
        agent = SACContinuous(
            env.observation_space.shape[0], hidden_dim,
            env.action_space.shape[0], action_bound,
            actor_lr=3e-4, critic_lr=3e-3, alpha_lr=3e-4,
            target_entropy=-env.action_space.shape[0], tau=tau, gamma=0.99,
            device=device)
        buffer_size, minimal_size = 100000, 1000
        tag = "ch14_sac_pendulum.png"
    else:
        env_name = "CartPole-v1"
        env = gym.make(env_name)
        episodes = args.episodes if args.episodes is not None else 200
        agent = SAC(env.observation_space.shape[0], hidden_dim,
                    int(env.action_space.n), actor_lr=1e-3, critic_lr=1e-2,
                    alpha_lr=1e-2, target_entropy=-1, tau=tau, gamma=0.98,
                    device=device)
        buffer_size, minimal_size = 10000, 500
        tag = "ch14_sac_cartpole.png"

    env.reset(seed=args.seed)
    replay_buffer = ReplayBuffer(buffer_size)
    return_list = train_off_policy_agent(env, agent, episodes, replay_buffer,
                                         minimal_size, batch_size,
                                         seed=args.seed)
    print(f"最后 20 回合平均回报: {np.mean(return_list[-20:]):.3f}")
    path = plot_returns(return_list, f"SAC on {env_name}", tag)
    print(f"图像已保存: {path}")
    env.close()


if __name__ == "__main__":
    main()
