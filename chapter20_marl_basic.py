"""第 20 章 多智能体强化学习入门（IPPO）

书中示例：在 15×15 的 Combat 格子世界上，用「参数共享 PPO」训练 2 个协作智能体
对抗 2 个敌方智能体（IPPO = Independent PPO）。

【环境说明】书中的 Combat 来自外部仓库 `ma-gym`（依赖旧版 gym，且需联网 clone）。
本脚本用一个与之等价的简化版 Combat 环境替代：15×15 格子、2v2、生命值 3、
5 个离散动作（上/下/左/右/攻击），观测包含自己、队友和两个敌人的相对信息，
击败全部敌人即获胜。算法与训练流程保持与书中一致。

运行（需要 torch）：
    python chapter20_marl_basic.py --episodes 500
    python chapter20_marl_basic.py --episodes 200 --grid 9
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from hrl.plotting import plt, save_fig
from hrl.rl_utils import compute_advantage, get_device

# 动作编号：0 上、1 下、2 左、3 右、4 攻击
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))


class CombatLite:
    """简化的 2v2 Combat 环境（接口模仿 ma-gym：观测/动作空间都是列表）。"""

    def __init__(self, grid_shape=(15, 15), n_agents=2, n_opponents=2,
                 max_steps=100, health=3, seed=None):
        self.nrow, self.ncol = grid_shape
        self.n_agents = n_agents
        self.n_opponents = n_opponents
        self.max_steps = max_steps
        self.health = health
        self.agents = [f"agent_{i}" for i in range(n_agents)]
        self.opponents = [f"opponent_{i}" for i in range(n_opponents)]
        self.rng = np.random.default_rng(seed)
        # 每个观测：自己(3) + 队友(3) + 两个敌人(2*3) = 12 维
        obs_dim = 3 + 3 * (n_agents - 1) + 3 * n_opponents
        self.observation_space = [gym.spaces.Box(-1.0, 1.0, (obs_dim,),
                                                dtype=np.float32)
                                  for _ in range(n_agents)]
        self.action_space = [gym.spaces.Discrete(5) for _ in range(n_agents)]

    # ------------------------------------------------------------------ 工具
    def _spawn(self, team: str, i: int):
        if team == "agent":
            x = self.rng.integers(0, 3)
        else:
            x = self.rng.integers(self.ncol - 3, self.ncol)
        y = self.rng.integers(0, self.nrow)
        return [int(x), int(y), self.health]

    def _obs(self, me, allies, enemies):
        """构造单个智能体的观测：归一化坐标与生命值。"""
        def rel(other):
            return [(other[0] - me[0]) / self.ncol,
                    (other[1] - me[1]) / self.nrow,
                    other[2] / self.health]
        own = [me[0] / self.ncol, me[1] / self.nrow, me[2] / self.health]
        parts = own + [v for a in allies for v in rel(a)]
        parts += [v for e in enemies for v in rel(e)]
        return np.asarray(parts, dtype=np.float32)

    def _attack(self, attacker, targets) -> int:
        """攻击：命中范围内（且存活）的第一个敌人，返回造成的伤害。"""
        for t in targets:
            if t[2] > 0 and max(abs(t[0] - attacker[0]),
                                abs(t[1] - attacker[1])) <= 1:
                t[2] -= 1
                return 1
        return 0

    # ------------------------------------------------------------------ 交互
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agent_states = [self._spawn("agent", i)
                             for i in range(self.n_agents)]
        self.opponent_states = [self._spawn("opponent", i)
                                for i in range(self.n_opponents)]
        self.step_count = 0
        return [self._obs(self.agent_states[i],
                          [s for j, s in enumerate(self.agent_states) if j != i],
                          self.opponent_states)
                for i in range(self.n_agents)]

    def step(self, actions):
        self.step_count += 1
        rewards = [0.0] * self.n_agents

        # 先移动（攻击动作不移动）
        for i, a in enumerate(actions):
            if a < 4:
                dx, dy = MOVES[a]
                self.agent_states[i][0] = int(np.clip(
                    self.agent_states[i][0] + dx, 0, self.ncol - 1))
                self.agent_states[i][1] = int(np.clip(
                    self.agent_states[i][1] + dy, 0, self.nrow - 1))

        # 敌方策略：随机移动，靠近了就攻击
        for j, opp in enumerate(self.opponent_states):
            if opp[2] <= 0:
                continue
            target_idx = min(
                range(len(self.agent_states)),
                key=lambda k: abs(self.agent_states[k][0] - opp[0]) +
                abs(self.agent_states[k][1] - opp[1]))
            target = self.agent_states[target_idx]
            if max(abs(target[0] - opp[0]), abs(target[1] - opp[1])) <= 1:
                if target[2] > 0:
                    target[2] -= 1
                    rewards[target_idx] -= 1.0
            else:
                dx = int(np.sign(target[0] - opp[0]))
                dy = int(np.sign(target[1] - opp[1]))
                if self.rng.random() < 0.5:
                    opp[0] = int(np.clip(opp[0] + dx, 0, self.ncol - 1))
                else:
                    opp[1] = int(np.clip(opp[1] + dy, 0, self.nrow - 1))

        # 智能体攻击
        for i, a in enumerate(actions):
            if a == 4:
                hit = self._attack(self.agent_states[i], self.opponent_states)
                rewards[i] += 1.0 * hit

        win = all(o[2] <= 0 for o in self.opponent_states)
        lose = all(s[2] <= 0 for s in self.agent_states)
        done = win or lose or self.step_count >= self.max_steps
        dones = [done] * self.n_agents
        next_obs = [self._obs(self.agent_states[i],
                              [s for j, s in enumerate(self.agent_states)
                               if j != i],
                              self.opponent_states)
                    for i in range(self.n_agents)]
        return next_obs, rewards, dones, {"win": bool(win), "lose": bool(lose)}


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return F.softmax(self.fc3(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.fc3(x)


class PPO:
    """PPO（截断目标）；两个智能体共享同一套网络参数。"""

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
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

        log_probs = torch.log(self.actor(states).gather(1, actions))
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
    parser = argparse.ArgumentParser(description="第 20 章 多智能体强化学习入门")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="书中设置是 100000，这里默认缩小")
    parser.add_argument("--grid", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    hidden_dim = 64
    gamma = 0.99
    lmbda = 0.97
    eps = 0.2
    device = get_device(args.cuda)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = CombatLite(grid_shape=(args.grid, args.grid), n_agents=2,
                     n_opponents=2, seed=args.seed)
    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    # 两个智能体共享同一个策略
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr=3e-4,
                critic_lr=1e-3, lmbda=lmbda, eps=eps, gamma=gamma, device=device)

    win_list = []
    pbar = tqdm(total=args.episodes, ncols=80, disable=None, desc="train")
    for i_episode in range(args.episodes):
        transitions = [
            {"states": [], "actions": [], "next_states": [],
             "rewards": [], "dones": []}
            for _ in range(2)
        ]
        s = env.reset()
        terminal = False
        while not terminal:
            a_1 = agent.take_action(s[0])
            a_2 = agent.take_action(s[1])
            next_s, r, done, info = env.step([a_1, a_2])
            for k, (a, sk, nsk) in enumerate(zip((a_1, a_2), s, next_s)):
                transitions[k]["states"].append(sk)
                transitions[k]["actions"].append(a)
                transitions[k]["next_states"].append(nsk)
                # 赢一局给 +100 奖励，否则每步 -0.1（与书中一致）
                transitions[k]["rewards"].append(
                    r[k] + 100 if info["win"] else r[k] - 0.1)
                transitions[k]["dones"].append(False)
            s = next_s
            terminal = all(done)
        win_list.append(1 if info["win"] else 0)
        agent.update(transitions[0])
        agent.update(transitions[1])
        if (i_episode + 1) % 20 == 0:
            pbar.set_postfix({"win(20)": "%.2f" % np.mean(win_list[-20:])})
        pbar.update(1)
    pbar.close()

    print(f"最后 100 局胜率: {np.mean(win_list[-100:]):.3f}")
    win_array = np.array(win_list)
    n = (len(win_array) // 50) * 50
    win_array = np.mean(win_array[:n].reshape(-1, 50), axis=1)  # 每 50 局取平均
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(win_array.shape[0]) * 50, win_array)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Win rate")
    ax.set_title("IPPO on Combat")
    path = save_fig(fig, "ch20_ippo_combat.png")
    print(f"图像已保存: {path}")


if __name__ == "__main__":
    main()
