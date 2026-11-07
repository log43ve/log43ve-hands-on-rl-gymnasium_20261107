"""第 9 章 策略梯度算法（REINFORCE）

对应书中示例：在 CartPole 上用蒙特卡洛策略梯度训练。

运行（需要 torch）：
    python chapter09_reinforce.py
    python chapter09_reinforce.py --episodes 200
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from hrl.plotting import plot_returns
from hrl.rl_utils import get_device


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class REINFORCE:
    """REINFORCE：用整条轨迹的回报估计策略梯度。"""

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate,
                 gamma, device):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma
        self.device = device

    def take_action(self, state) -> int:
        state = torch.as_tensor(np.array([state], dtype=np.float32), device=self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        return int(action_dist.sample().item())

    def update(self, transition_dict) -> None:
        reward_list = transition_dict["rewards"]
        state_list = transition_dict["states"]
        action_list = transition_dict["actions"]

        G = 0.0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = torch.tensor([state_list[i]],
                                 dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G  # 每一步的损失函数
            loss.backward()
        self.optimizer.step()


def train(env, agent, num_episodes: int, seed: int = 0):
    from tqdm import tqdm

    return_list = []
    state, info = env.reset(seed=seed)
    pbar = tqdm(total=num_episodes, ncols=80, disable=None, desc="train")
    for i_episode in range(num_episodes):
        if i_episode > 0:
            state, info = env.reset()
        episode_return = 0.0
        transition_dict = {"states": [], "actions": [], "next_states": [],
                           "rewards": [], "dones": []}
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            transition_dict["states"].append(state)
            transition_dict["actions"].append(action)
            transition_dict["next_states"].append(next_state)
            transition_dict["rewards"].append(reward)
            transition_dict["dones"].append(done)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
        agent.update(transition_dict)
        if (i_episode + 1) % 10 == 0:
            pbar.set_postfix({"return": "%.3f" % np.mean(return_list[-10:])})
        pbar.update(1)
    pbar.close()
    return return_list


def main() -> None:
    parser = argparse.ArgumentParser(description="第 9 章 策略梯度算法")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
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
    agent = REINFORCE(state_dim, hidden_dim, action_dim, args.lr, args.gamma,
                      device)

    return_list = train(env, agent, args.episodes, seed=args.seed)
    print(f"最后 20 回合平均回报: {np.mean(return_list[-20:]):.3f}")
    path = plot_returns(return_list, f"REINFORCE on {env_name}",
                        "ch09_reinforce.png")
    print(f"图像已保存: {path}")
    env.close()


if __name__ == "__main__":
    main()
