"""第 19 章 目标导向的强化学习（DDPG + HER）

对应书中示例：在一个自制的二维网格世界 ``WorldEnv`` 里，用 DDPG 训练一个
"把当前位置移动到目标点"的策略，并比较 Hindsight Experience Replay（HER）
与不使用 HER 的普通 DDPG 的差别。

世界设定（与书中一致）：
  * 状态是二维坐标，限制在 [0, 5] x [0, 5]；
  * 动作是 [-1, 1]^2 的位移，环境内部会把动作和结果坐标都裁剪到允许范围；
  * 每回合开始时目标点随机落在 [3.5, 4.5]^2，初始位置为 (0, 0)；
  * 当前位置与目标距离 <= 0.15 视为到达，回合成功结束；否则最多走 50 步；
  * 每步奖励 -1，到达目标的那一步奖励 0（因此回报越接近 0 越好）。

运行（需要 torch）：
    python chapter19_goal_rl.py                     # 书中默认 2000 回合
    python chapter19_goal_rl.py --episodes 30       # 快速跑一小段
    python chapter19_goal_rl.py --skip-baseline     # 只跑 HER，不跑对照组
"""

from __future__ import annotations

import argparse
import collections
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from tqdm import tqdm

from hrl.plotting import plot_returns
from hrl.rl_utils import get_device


class WorldEnv(gym.Env):
    """书中自制的二维网格世界（这里改写成 ``gymnasium.Env`` 子类）。

    观测是 4 维：``[state_x, state_y, goal_x, goal_y]``，与书中
    ``np.hstack((self.state, self.goal))`` 的布局完全一致，
    因此回放池和 HER 重新标注目标的代码可以直接沿用。
    """

    metadata = {"render_modes": []}  # 这些示例不需要渲染

    def __init__(self, distance_threshold: float = 0.15,
                 action_bound: float = 1.0, max_steps: int = 50):
        super().__init__()
        self.distance_threshold = distance_threshold
        self.action_bound = action_bound
        self.max_steps = max_steps
        # 动作是二维位移，两个维度都限制在 [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,),
                                       dtype=np.float32)
        # 观测 = 当前位置(2) + 目标位置(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(4,), dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.state = np.zeros(2, dtype=np.float32)
        self.count = 0

    def _get_obs(self) -> np.ndarray:
        return np.hstack((self.state, self.goal)).astype(np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        # 让 gymnasium 帮我们初始化 self.np_random（按 seed 可复现）
        super().reset(seed=seed)
        # 目标点坐标范围是 [3.5, 4.5] x [3.5, 4.5]
        self.goal = (4.0 + self.np_random.uniform(-0.5, 0.5, size=2)).astype(
            np.float32)
        self.state = np.zeros(2, dtype=np.float32)  # 初始状态
        self.count = 0
        info = {
            "goal": self.goal.copy(),
            "distance": float(np.linalg.norm(self.state - self.goal)),
            "is_success": False,
        }
        return self._get_obs(), info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32),
                         -self.action_bound, self.action_bound)
        # 位置被裁剪到 [0, 5] x [0, 5]
        x = float(np.clip(self.state[0] + action[0], 0.0, 5.0))
        y = float(np.clip(self.state[1] + action[1], 0.0, 5.0))
        self.state = np.array([x, y], dtype=np.float32)
        self.count += 1

        dis = float(np.linalg.norm(self.state - self.goal))
        reward = -1.0 if dis > self.distance_threshold else 0.0
        terminated = dis <= self.distance_threshold
        # 书里用 done 同时表示"成功"和"到达 50 步上限"，
        # 这里按 gymnasium 的约定拆成 terminated / truncated
        truncated = (not terminated) and self.count >= self.max_steps
        info = {
            "goal": self.goal.copy(),
            "distance": dis,
            "is_success": terminated,
        }
        return self._get_obs(), reward, terminated, truncated, info


class PolicyNet(torch.nn.Module):
    """确定性策略网络：输出经 tanh 缩放到 [-action_bound, action_bound]。"""

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int,
                 action_bound: float):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # 环境能接受的动作最大值

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return torch.tanh(self.fc3(x)) * self.action_bound


class QValueNet(torch.nn.Module):
    """动作价值网络 Q(s, a)。"""

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)  # 拼接状态和动作
        x = F.relu(self.fc2(F.relu(self.fc1(cat))))
        return self.fc3(x)


class DDPG:
    """DDPG 算法。"""

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int,
                 action_bound: float, actor_lr: float, critic_lr: float,
                 sigma: float, tau: float, gamma: float, device):
        self.action_dim = action_dim
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim,
                               action_bound).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim,
                                      action_bound).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim,
                                       action_dim).to(device)
        # 初始化目标价值网络并使其参数和价值网络一样
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并使其参数和策略网络一样
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差，均值直接设为 0
        self.tau = tau  # 目标网络软更新参数
        self.action_bound = action_bound
        self.device = device

    def take_action(self, state):
        # 用 as_tensor 避免 torch.tensor([ndarray]) 的性能警告
        state = torch.as_tensor(np.array([state], dtype=np.float32),
                                device=self.device)
        action = self.actor(state).detach().cpu().numpy()[0]
        # 给动作添加噪声，增加探索
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def soft_update(self, net, target_net) -> None:
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict) -> None:
        states = torch.as_tensor(transition_dict["states"],
                                 dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(transition_dict["actions"],
                                  dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(transition_dict["rewards"],
                                  dtype=torch.float32,
                                  device=self.device).view(-1, 1)
        next_states = torch.as_tensor(transition_dict["next_states"],
                                      dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(transition_dict["dones"], dtype=torch.float32,
                                device=self.device).view(-1, 1)

        next_q_values = self.target_critic(next_states,
                                           self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        # MSE 损失函数
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 策略网络就是为了使 Q 值最大化
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络


class Trajectory:
    """用来记录一条完整轨迹（HER 需要轨迹里的所有状态）。"""

    def __init__(self, init_state):
        self.states = [init_state]
        self.actions = []
        self.rewards = []
        self.dones = []
        self.length = 0

    def store_step(self, action, state, reward, done) -> None:
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.length += 1


class ReplayBuffer_Trajectory:
    """存储轨迹的经验回放池（书中第 19 章专用）。"""

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def add_trajectory(self, trajectory: Trajectory) -> None:
        self.buffer.append(trajectory)

    def size(self) -> int:
        return len(self.buffer)

    def sample(self, batch_size: int, use_her: bool,
               dis_threshold: float = 0.15, her_ratio: float = 0.8):
        batch = dict(states=[], actions=[], next_states=[], rewards=[],
                     dones=[])
        for _ in range(batch_size):
            traj = random.sample(self.buffer, 1)[0]
            step_state = np.random.randint(traj.length)
            state = traj.states[step_state]
            next_state = traj.states[step_state + 1]
            action = traj.actions[step_state]
            reward = traj.rewards[step_state]
            done = traj.dones[step_state]

            if use_her and np.random.uniform() <= her_ratio:
                # 从同一个回合的"未来"状态里挑一个当作目标
                step_goal = np.random.randint(step_state + 1, traj.length + 1)
                goal = traj.states[step_goal][:2]
                dis = np.sqrt(np.sum(np.square(next_state[:2] - goal)))
                reward = -1.0 if dis > dis_threshold else 0
                done = False if dis > dis_threshold else True
                state = np.hstack((state[:2], goal))
                next_state = np.hstack((next_state[:2], goal))

            batch["states"].append(state)
            batch["next_states"].append(next_state)
            batch["actions"].append(action)
            batch["rewards"].append(reward)
            batch["dones"].append(done)

        batch["states"] = np.array(batch["states"])
        batch["next_states"] = np.array(batch["next_states"])
        batch["actions"] = np.array(batch["actions"])
        return batch


def set_seed(seed: int) -> None:
    """固定 Python / NumPy / PyTorch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(env: WorldEnv, agent: DDPG, replay_buffer,
          num_episodes: int, use_her: bool, minimal_episodes: int,
          n_train: int, batch_size: int, seed: int):
    """训练并返回 (每回合回报, 每回合是否成功)。"""
    env.reset(seed=seed)

    return_list = []
    success_list = []
    pbar = tqdm(total=num_episodes, ncols=80, disable=None,
                desc="HER" if use_her else "no-HER")
    for i_episode in range(num_episodes):
        state, info = env.reset()
        traj = Trajectory(state)
        episode_return = 0.0
        done = False
        success = False
        while not done:
            action = agent.take_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            success = terminated
            episode_return += reward
            traj.store_step(action, state, reward, done)
        replay_buffer.add_trajectory(traj)
        return_list.append(episode_return)
        success_list.append(float(success))
        if replay_buffer.size() >= minimal_episodes:
            for _ in range(n_train):
                transition_dict = replay_buffer.sample(batch_size, use_her)
                agent.update(transition_dict)
        if (i_episode + 1) % 10 == 0:
            pbar.set_postfix({
                "return": "%.3f" % np.mean(return_list[-10:]),
                "success": "%.2f" % np.mean(success_list[-10:]),
            })
        pbar.update(1)
    pbar.close()
    return return_list, success_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="第 19 章 目标导向的强化学习（DDPG + HER）")
    parser.add_argument("--episodes", type=int, default=2000,
                        help="训练回合数（书中默认 2000）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true", help="有 GPU 时使用 GPU")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="跳过不使用 HER 的对照组")
    args = parser.parse_args()

    actor_lr = 1e-3
    critic_lr = 1e-3
    hidden_dim = 128
    state_dim = 4  # [state_x, state_y, goal_x, goal_y]
    action_dim = 2
    action_bound = 1.0
    sigma = 0.1
    tau = 0.005
    gamma = 0.98
    n_train = 20
    batch_size = 256
    buffer_size = 10000
    # 书里固定为 200：不够 200 条轨迹就不训练。为了让 --episodes 较小的
    # 快速实验也能看到学习效果，这里按回合数把预热轨迹数自动缩小；
    # 当 --episodes 取书中的 2000 时恰好还是 200。
    minimal_episodes = min(200, max(10, args.episodes // 10))

    device = get_device(args.cuda)
    # 这些网络的张量都很小，多线程反而会被线程同步拖慢（CPU 上约 2ms/次更新）
    if device.type == "cpu":
        torch.set_num_threads(1)
    print(f"设备: {device} | 回合数: {args.episodes} | "
          f"预热轨迹数: {minimal_episodes}")

    env = WorldEnv()

    # ---------- 实验 1：DDPG + HER（书中的主实验） ----------
    set_seed(args.seed)
    replay_buffer = ReplayBuffer_Trajectory(buffer_size)
    agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, actor_lr,
                 critic_lr, sigma, tau, gamma, device)
    return_list, success_list = train(env, agent, replay_buffer, args.episodes,
                                      True, minimal_episodes, n_train,
                                      batch_size, args.seed)
    avg_return = float(np.mean(return_list[-20:]))
    success_rate = float(np.mean(success_list[-20:]))
    print(f"[DDPG + HER] 最后 {min(20, len(return_list))} 回合平均回报: "
          f"{avg_return:.3f} | 成功率: {success_rate:.2%}")
    path = plot_returns(
        return_list, "DDPG with HER on GridWorld", "ch19_ddpg_her.png",
        ylabel="Returns")
    print(f"图像已保存: {path}")

    # ---------- 实验 2：不使用 HER 的对照组 ----------
    if not args.skip_baseline:
        set_seed(args.seed)
        replay_buffer = ReplayBuffer_Trajectory(buffer_size)
        agent = DDPG(state_dim, hidden_dim, action_dim, action_bound,
                     actor_lr, critic_lr, sigma, tau, gamma, device)
        return_list, success_list = train(env, agent, replay_buffer,
                                          args.episodes, False,
                                          minimal_episodes, n_train,
                                          batch_size, args.seed)
        avg_return = float(np.mean(return_list[-20:]))
        success_rate = float(np.mean(success_list[-20:]))
        print(f"[DDPG] 最后 {min(20, len(return_list))} 回合平均回报: "
              f"{avg_return:.3f} | 成功率: {success_rate:.2%}")
        path = plot_returns(
            return_list, "DDPG without HER on GridWorld",
            "ch19_ddpg_no_her.png", ylabel="Returns")
        print(f"图像已保存: {path}")

    env.close()


if __name__ == "__main__":
    main()
