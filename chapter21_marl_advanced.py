"""第 21 章 多智能体强化学习进阶（MADDPG）

书中示例：在 simple_adversary 场景上训练 MADDPG —— 1 个「捣乱者」与 2 个
协作智能体争夺同一个地标，用集中的 Critic 解决多智能体环境非平稳的问题。

【环境说明】书中的 simple_adversary 来自外部仓库 `multiagent-particle-envs`
（依赖旧版 gym，需联网 clone 并降级 gym）。本脚本内置一个等价的简化场景：
  * 3 个智能体：agents[0] 是捣乱者（adversary），agents[1]、agents[2] 是协作智能体
  * 二维平面，5 个离散动作（不动/上/下/左/右），每步移动 0.1
  * 地标位置随机；协作智能体的观测包含地标位置，捣乱者的观测不包含
  * 奖励：协作智能体 = -自身到地标的距离；捣乱者 = -自己到地标的距离（抢占地标）
  * 每条序列 25 步，动作是 one-hot，和书中 MADDPG 的离散动作版本一致

运行（需要 torch）：
    python chapter21_marl_advanced.py --episodes 300
"""

from __future__ import annotations

import argparse
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from hrl.plotting import moving_average, plt, save_fig
from hrl.rl_utils import ReplayBuffer, get_device

MOVES = ((-0.1, 0.0), (0.1, 0.0), (0.0, -0.1), (0.0, 0.1))


class SimpleAdversaryLite:
    """简化版 simple_adversary：1 个捣乱者 + 2 个协作智能体 + 1 个地标。"""

    def __init__(self, episode_length: int = 25, seed: int | None = None):
        self.n_agents = 3
        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed)
        self.agents = ["adversary_0", "agent_0", "agent_1"]
        self.action_space = [gym.spaces.Discrete(5) for _ in range(self.n_agents)]
        # 观测：自己的位置(2) + 其他智能体位置(2*2) + 地标位置(2) = 8 维
        self.observation_space = [
            gym.spaces.Box(-1.0, 1.0, (8,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        self.landmark = None
        self.positions = None
        self.t = 0

    def _obs(self) -> list[np.ndarray]:
        obs = []
        for i in range(self.n_agents):
            others = [self.positions[j] for j in range(self.n_agents) if j != i]
            parts = list(self.positions[i]) + [v for o in others for v in o]
            # 捣乱者看不到地标位置（用 0 填充），协作智能体可以看到
            parts += list(self.landmark) if i != 0 else [0.0, 0.0]
            obs.append(np.asarray(parts, dtype=np.float32))
        return obs

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.landmark = self.rng.uniform(-1.0, 1.0, size=2)
        self.positions = [self.rng.uniform(-1.0, 1.0, size=2)
                          for _ in range(self.n_agents)]
        self.t = 0
        return self._obs()

    def step(self, actions):
        self.t += 1
        for i, a in enumerate(actions):
            a = int(a)
            if a > 0:  # 0 表示不动
                self.positions[i] = np.clip(
                    self.positions[i] + np.array(MOVES[a - 1]), -1.0, 1.0)

        dist_good = np.mean([np.linalg.norm(self.positions[i] - self.landmark)
                             for i in (1, 2)])
        rew = [-np.linalg.norm(self.positions[0] - self.landmark), -dist_good,
               -dist_good]
        done = self.t >= self.episode_length
        return self._obs(), rew, [done] * self.n_agents, {}


def onehot_from_logits(logits, eps: float = 0.01):
    """把 logits 变成 one-hot，并以 eps 概率随机探索。"""
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    rand_acs = torch.eye(logits.shape[1])[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ].to(logits.device)
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])


def sample_gumbel(shape, eps: float = 1e-20, tens_type=torch.FloatTensor):
    U = torch.empty(*shape, dtype=tens_type().dtype).uniform_()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature: float = 1.0):
    y = logits + sample_gumbel(logits.shape).to(logits.device)
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature: float = 1.0):
    """Gumbel-Softmax：前向是 one-hot，反向可导。"""
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)
    return (y_hard.to(logits.device) - y).detach() + y


class TwoLayerFC(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDPG:
    """单个智能体的 actor-critic（动作是离散 one-hot）。"""

    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device):
        self.actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)
        self.target_actor = TwoLayerFC(state_dim, action_dim,
                                       hidden_dim).to(device)
        self.critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)
        self.target_critic = TwoLayerFC(critic_input_dim, 1,
                                        hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)

    def take_action(self, state, explore: bool = False):
        action = self.actor(state)
        action = gumbel_softmax(action) if explore else onehot_from_logits(action)
        return action.detach().cpu().numpy()[0]

    def soft_update(self, net, target_net, tau: float) -> None:
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)


class MADDPG:
    """MADDPG：每个智能体一个 actor，Critic 输入所有智能体的观测与动作。"""

    def __init__(self, n_agents, device, actor_lr, critic_lr, hidden_dim,
                 state_dims, action_dims, critic_input_dim, gamma, tau):
        self.n_agents = n_agents
        self.agents = [
            DDPG(state_dims[i], action_dims[i], critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device)
            for i in range(n_agents)
        ]
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]

    def take_action(self, states, explore: bool):
        states = [torch.as_tensor(np.array([states[i]], dtype=np.float32),
                                  device=self.device)
                  for i in range(self.n_agents)]
        return [agent.take_action(state, explore)
                for agent, state in zip(self.agents, states)]

    def update(self, sample, i_agent: int) -> None:
        obs, act, rew, next_obs, done = sample
        cur_agent = self.agents[i_agent]

        # 更新 Critic：目标动作由所有目标策略给出
        cur_agent.critic_optimizer.zero_grad()
        all_target_act = [onehot_from_logits(pi(_next_obs))
                          for pi, _next_obs in zip(self.target_policies, next_obs)]
        target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)
        target_critic_value = rew[i_agent].view(-1, 1) + self.gamma * \
            cur_agent.target_critic(target_critic_input) * \
            (1 - done[i_agent].view(-1, 1))
        critic_input = torch.cat((*obs, *act), dim=1)
        critic_value = cur_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value,
                                            target_critic_value.detach())
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        # 更新 Actor：用当前策略的 Gumbel-Softmax 动作作为 Critic 输入
        cur_agent.actor_optimizer.zero_grad()
        cur_actor_out = cur_agent.actor(obs[i_agent])
        cur_act_vf_in = gumbel_softmax(cur_actor_out)
        all_actor_acs = []
        for i, (pi, _obs) in enumerate(zip(self.policies, obs)):
            if i == i_agent:
                all_actor_acs.append(cur_act_vf_in)
            else:
                all_actor_acs.append(onehot_from_logits(pi(_obs)))
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        actor_loss = -cur_agent.critic(vf_in).mean()
        actor_loss += (cur_actor_out ** 2).mean() * 1e-3
        actor_loss.backward()
        cur_agent.actor_optimizer.step()

    def update_all_targets(self) -> None:
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)


def stack_array(x):
    """把 (batch, n_agents, dim) 的数组拆成每个智能体一个 (batch, dim) tensor。"""
    rearranged = [[sub_x[i] for sub_x in x] for i in range(len(x[0]))]
    return [torch.FloatTensor(np.vstack(aa)) for aa in rearranged]


def evaluate(maddpg, n_episode: int = 10, episode_length: int = 25):
    env = SimpleAdversaryLite(episode_length=episode_length)
    returns = np.zeros(env.n_agents)
    for _ in range(n_episode):
        obs = env.reset()
        for _ in range(episode_length):
            actions = maddpg.take_action(obs, explore=False)
            obs, rew, done, info = env.step(
                [int(np.argmax(a)) for a in actions])
            returns += np.array(rew) / n_episode
    return returns.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="第 21 章 MADDPG")
    parser.add_argument("--episodes", type=int, default=500,
                        help="书中设置是 5000，这里默认缩小")
    parser.add_argument("--episode-length", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    hidden_dim = 64
    actor_lr = 1e-2
    critic_lr = 1e-2
    gamma = 0.95
    tau = 1e-2
    buffer_size = 100000
    minimal_size = 4000
    update_interval = 100

    device = get_device(args.cuda)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = SimpleAdversaryLite(episode_length=args.episode_length,
                              seed=args.seed)
    state_dims = [sp.shape[0] for sp in env.observation_space]
    action_dims = [sp.n for sp in env.action_space]
    critic_input_dim = sum(state_dims) + sum(action_dims)

    maddpg = MADDPG(len(env.agents), device, actor_lr, critic_lr, hidden_dim,
                    state_dims, action_dims, critic_input_dim, gamma, tau)
    replay_buffer = ReplayBuffer(buffer_size)

    return_list = []
    total_step = 0
    for i_episode in range(args.episodes):
        state = env.reset()
        for _ in range(args.episode_length):
            actions = maddpg.take_action(state, explore=True)
            next_state, reward, done, _ = env.step(
                [int(np.argmax(a)) for a in actions])
            replay_buffer.add(state, actions, reward, next_state, done)
            state = next_state
            total_step += 1
            if (replay_buffer.size() >= minimal_size and
                    total_step % update_interval == 0):
                sample = replay_buffer.sample(args.batch_size)
                sample = [stack_array(x) for x in sample]
                for a_i in range(len(env.agents)):
                    maddpg.update(sample, a_i)
                maddpg.update_all_targets()
        if (i_episode + 1) % 50 == 0:
            ep_returns = evaluate(maddpg, n_episode=10,
                                  episode_length=args.episode_length)
            return_list.append(ep_returns)
            print(f"Episode: {i_episode + 1}, {np.round(ep_returns, 2)}")

    return_array = np.array(return_list)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, agent_name in enumerate(env.agents):
        ax.plot(np.arange(return_array.shape[0]) * 50,
                moving_average(return_array[:, i], 5), label=agent_name)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Returns")
    ax.set_title("MADDPG on SimpleAdversary")
    ax.legend()
    path = save_fig(fig, "ch21_maddpg_simple_adversary.png")
    print(f"图像已保存: {path}")


if __name__ == "__main__":
    main()
