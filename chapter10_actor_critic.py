"""第 10 章 Actor-Critic 算法

对应书中示例：在 CartPole 上用 Actor-Critic（策略网络 + 价值网络）训练。

运行（需要 torch）：
    python chapter10_actor_critic.py
    python chapter10_actor_critic.py --episodes 300
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from hrl.plotting import plot_returns
from hrl.rl_utils import get_device, train_on_policy_agent


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


class ActorCritic:
    """Actor-Critic：用 TD 误差同时更新策略网络和价值网络。"""

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.device = device

    def take_action(self, state) -> int:
        state = torch.as_tensor(np.array([state], dtype=np.float32), device=self.device)
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

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)  # 时序差分误差
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()


def main() -> None:
    parser = argparse.ArgumentParser(description="第 10 章 Actor-Critic 算法")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-2)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    hidden_dim = 128
    device = get_device(args.cuda)

    env_name = "CartPole-v1"
    env = gym.make(env_name)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.reset(seed=args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = int(env.action_space.n)
    agent = ActorCritic(state_dim, hidden_dim, action_dim, args.actor_lr,
                        args.critic_lr, args.gamma, device)

    return_list = train_on_policy_agent(env, agent, args.episodes,
                                        seed=args.seed)
    print(f"最后 20 回合平均回报: {np.mean(return_list[-20:]):.3f}")
    path = plot_returns(return_list, f"Actor-Critic on {env_name}",
                        "ch10_actor_critic.png")
    print(f"图像已保存: {path}")
    env.close()


if __name__ == "__main__":
    main()
