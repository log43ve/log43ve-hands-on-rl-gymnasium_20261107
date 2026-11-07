"""第 15 章 模仿学习：行为克隆（BC）与生成对抗模仿学习（GAIL）

对应书中示例：先在 CartPole 上用 PPO 训练出一个“专家”，只采集少量专家轨迹，
然后分别用 BC（对专家动作做最大似然估计）和 GAIL（判别器给出奖励 + PPO 更新策略）
去模仿专家策略。

运行（需要 torch）：
    python chapter15_imitation.py
    python chapter15_imitation.py --episodes 30 --ppo-episodes 20 --bc-iterations 200

默认超参数与书中一致：PPO 250 回合、专家数据 1 条轨迹、BC 1000 次迭代、
GAIL 500 回合。注意 gymnasium 中 CartPole 只有 v1（v0 已移除），单回合上限
500 步，因此回报上限是 500 而不是书中的 200。
"""

from __future__ import annotations

import argparse
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from hrl.plotting import plot_returns, plt, save_fig
from hrl.rl_utils import compute_advantage, get_device, train_on_policy_agent


class PolicyNet(torch.nn.Module):
    """策略网络：状态 -> 动作概率分布。"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    """价值网络：状态 -> 状态价值。"""

    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    """PPO 算法，采用截断方式（第 12 章的策略，这里用来训练专家策略和 GAIL 的策略）。"""

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
        self.epochs = epochs  # 一条序列的数据用于训练轮数
        self.eps = eps  # PPO 中截断范围的参数
        self.device = device

    def take_action(self, state) -> int:
        # 用 as_tensor 构造单条观测，避免 torch.tensor([state]) 的告警
        state = torch.as_tensor(np.array([state], dtype=np.float32),
                                device=self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        return int(action_dist.sample().item())

    def update(self, transition_dict) -> None:
        # 先用 numpy 把列表拼成数组，再转成 tensor（比 list->tensor 快很多）
        states = torch.as_tensor(
            np.array(transition_dict['states'], dtype=np.float32),
            device=self.device)
        actions = torch.as_tensor(
            np.array(transition_dict['actions']),
            device=self.device).view(-1, 1)
        rewards = torch.as_tensor(
            np.array(transition_dict['rewards'], dtype=np.float32),
            device=self.device).view(-1, 1)
        next_states = torch.as_tensor(
            np.array(transition_dict['next_states'], dtype=np.float32),
            device=self.device)
        dones = torch.as_tensor(
            np.array(transition_dict['dones'], dtype=np.float32),
            device=self.device).view(-1, 1)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                      td_delta.cpu()).to(self.device)
        # 用 clamp 防止概率下溢导致 log(0) = -inf
        old_log_probs = torch.log(
            self.actor(states).gather(1, actions).clamp(min=1e-8)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(
                self.actor(states).gather(1, actions).clamp(min=1e-8))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO 损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


def sample_expert_data(env, agent, n_episode):
    """用专家策略（训练好的 PPO）采样若干条轨迹，返回 (状态, 动作)。"""
    states = []
    actions = []
    for episode in range(n_episode):
        state, info = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            states.append(state)
            actions.append(action)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
    return np.array(states), np.array(actions)


class BehaviorClone:
    """行为克隆：把专家动作当作标签，对策略网络做最大似然估计。"""

    def __init__(self, state_dim, hidden_dim, action_dim, lr, device):
        self.policy = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.device = device

    def learn(self, states, actions) -> None:
        states = torch.as_tensor(np.asarray(states, dtype=np.float32),
                                 device=self.device)
        actions = torch.as_tensor(np.asarray(actions), device=self.device)
        actions = actions.view(-1, 1)
        # 用 clamp 防止概率下溢导致 log(0) = -inf
        log_probs = torch.log(
            self.policy(states).gather(1, actions).clamp(min=1e-8))
        bc_loss = torch.mean(-log_probs)  # 最大似然估计

        self.optimizer.zero_grad()
        bc_loss.backward()
        self.optimizer.step()

    def take_action(self, state) -> int:
        state = torch.as_tensor(np.array([state], dtype=np.float32),
                                device=self.device)
        probs = self.policy(state)
        action_dist = torch.distributions.Categorical(probs)
        return int(action_dist.sample().item())


def test_agent(agent, env, n_episode):
    """跑 n_episode 个回合，返回平均回报。"""
    return_list = []
    for episode in range(n_episode):
        episode_return = 0
        state, info = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
    return float(np.mean(return_list))


class Discriminator(torch.nn.Module):
    """GAIL 的判别器：判断 (状态, 动作) 来自专家还是来自策略。"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        return torch.sigmoid(self.fc2(x))


class GAIL:
    """GAIL：判别器区分专家与策略数据，其输出被转换成奖励喂给 PPO。"""

    def __init__(self, agent, state_dim, action_dim, hidden_dim, lr_d, device):
        self.discriminator = Discriminator(state_dim, hidden_dim,
                                           action_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d)
        self.agent = agent
        self.action_dim = action_dim
        self.device = device

    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s, dones) -> None:
        expert_states = torch.as_tensor(np.asarray(expert_s, dtype=np.float32),
                                        device=self.device)
        expert_actions = torch.as_tensor(np.asarray(expert_a),
                                         device=self.device)
        agent_states = torch.as_tensor(np.asarray(agent_s, dtype=np.float32),
                                       device=self.device)
        agent_actions = torch.as_tensor(np.asarray(agent_a),
                                        device=self.device)
        expert_actions = F.one_hot(expert_actions,
                                   num_classes=self.action_dim).float()
        agent_actions = F.one_hot(agent_actions,
                                  num_classes=self.action_dim).float()

        expert_prob = self.discriminator(expert_states, expert_actions)
        agent_prob = self.discriminator(agent_states, agent_actions)
        discriminator_loss = nn.BCELoss()(
            agent_prob, torch.ones_like(agent_prob)) + nn.BCELoss()(
                expert_prob, torch.zeros_like(expert_prob))
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        # 策略“骗过”判别器的程度越高，得到的奖励越大
        rewards = -torch.log(agent_prob.clamp(min=1e-8)).detach().cpu().numpy()
        transition_dict = {
            'states': agent_s,
            'actions': agent_a,
            'rewards': rewards,
            'next_states': next_s,
            'dones': dones
        }
        self.agent.update(transition_dict)


def main() -> None:
    parser = argparse.ArgumentParser(description="第 15 章 模仿学习（BC + GAIL）")
    parser.add_argument("--episodes", type=int, default=500,
                        help="GAIL 训练回合数（书中 500）")
    parser.add_argument("--ppo-episodes", type=int, default=250,
                        help="训练专家策略的 PPO 回合数（书中 250）")
    parser.add_argument("--expert-episodes", type=int, default=1,
                        help="采集专家数据的轨迹条数（书中 1）")
    parser.add_argument("--bc-iterations", type=int, default=1000,
                        help="行为克隆的迭代次数（书中 1000）")
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-2)
    parser.add_argument("--lr-d", type=float, default=1e-3,
                        help="GAIL 判别器的学习率（书中 1e-3）")
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--lmbda", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=10,
                        help="PPO 每条序列的训练轮数（书中 10）")
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true", help="有 GPU 时使用 GPU")
    args = parser.parse_args()

    hidden_dim = 128
    device = get_device(args.cuda)

    env_name = "CartPole-v1"
    env = gym.make(env_name)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    env.reset(seed=args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = int(env.action_space.n)

    # ------------------------------------------------------------------
    # 第一步：用 PPO 训练一个专家策略
    # ------------------------------------------------------------------
    ppo_agent = PPO(state_dim, hidden_dim, action_dim, args.actor_lr,
                    args.critic_lr, args.lmbda, args.epochs, args.eps,
                    args.gamma, device)
    train_on_policy_agent(env, ppo_agent, args.ppo_episodes, seed=args.seed)

    # ------------------------------------------------------------------
    # 第二步：采集少量专家数据（书中只采样 1 条轨迹，再随机取 30 个样本）
    # ------------------------------------------------------------------
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    expert_s, expert_a = sample_expert_data(env, ppo_agent,
                                            args.expert_episodes)

    n_samples = 30  # 采样 30 个数据
    n_samples = min(n_samples, expert_s.shape[0])  # 轨迹太短时避免取样越界
    random_index = random.sample(range(expert_s.shape[0]), n_samples)
    expert_s = expert_s[random_index]
    expert_a = expert_a[random_index]

    # ------------------------------------------------------------------
    # 第三步：行为克隆（BC）
    # ------------------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    bc_agent = BehaviorClone(state_dim, hidden_dim, action_dim, args.actor_lr,
                             device)
    batch_size = 64
    test_returns = []

    with tqdm(total=args.bc_iterations, desc="BC", disable=None) as pbar:
        for i in range(args.bc_iterations):
            sample_indices = np.random.randint(low=0,
                                               high=expert_s.shape[0],
                                               size=batch_size)
            bc_agent.learn(expert_s[sample_indices], expert_a[sample_indices])
            current_return = test_agent(bc_agent, env, 5)
            test_returns.append(current_return)
            if (i + 1) % 10 == 0:
                pbar.set_postfix({'return': '%.3f' % np.mean(test_returns[-10:])})
            pbar.update(1)

    print(f"BC 最后 20 次迭代平均回报: {np.mean(test_returns[-20:]):.3f}")
    # 书中的 BC 曲线画的是“迭代次数 - 回报”
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(test_returns)), test_returns)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Returns")
    ax.set_title(f"BC on {env_name}")
    bc_path = save_fig(fig, "ch15_bc.png")
    print(f"图像已保存: {bc_path}")

    # ------------------------------------------------------------------
    # 第四步：生成对抗模仿学习（GAIL）
    # ------------------------------------------------------------------
    torch.manual_seed(args.seed)
    agent = PPO(state_dim, hidden_dim, action_dim, args.actor_lr,
                args.critic_lr, args.lmbda, args.epochs, args.eps, args.gamma,
                device)
    gail = GAIL(agent, state_dim, action_dim, hidden_dim, args.lr_d, device)
    return_list = []

    with tqdm(total=args.episodes, desc="GAIL", disable=None) as pbar:
        for i in range(args.episodes):
            episode_return = 0
            state, info = env.reset()
            done = False
            state_list = []
            action_list = []
            next_state_list = []
            done_list = []
            while not done:
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, info = env.step(
                    action)
                done = terminated or truncated
                state_list.append(state)
                action_list.append(action)
                next_state_list.append(next_state)
                done_list.append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            gail.learn(expert_s, expert_a, state_list, action_list,
                       next_state_list, done_list)
            if (i + 1) % 10 == 0:
                pbar.set_postfix({'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)

    print(f"最后 20 回合平均回报: {np.mean(return_list[-20:]):.3f}")
    path = plot_returns(return_list, f"GAIL on {env_name}", "ch15_gail.png")
    print(f"图像已保存: {path}")
    env.close()


if __name__ == "__main__":
    main()
