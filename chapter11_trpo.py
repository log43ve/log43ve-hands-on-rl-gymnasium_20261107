"""第 11 章 TRPO 算法

对应书中示例：
  * 离散动作（CartPole）上的 TRPO
  * 连续动作（Pendulum）上的 TRPO

运行（需要 torch）：
    python chapter11_trpo.py --env cartpole --episodes 200
    python chapter11_trpo.py --env pendulum --episodes 200   # 书中默认 2000

TRPO 用共轭梯度法求自然梯度方向，再做线性搜索保证 KL 约束，
因此这里没有使用优化器直接更新策略网络，而是手动更新网络参数。
"""

from __future__ import annotations

import argparse
import copy

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from hrl.plotting import plot_returns
from hrl.rl_utils import compute_advantage, get_device, train_on_policy_agent


class PolicyNet(torch.nn.Module):
    """离散动作策略网络。"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    """价值网络。"""

    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TRPO:
    """离散动作的 TRPO 算法。"""

    def __init__(self, hidden_dim, state_space, action_space, lmbda,
                 kl_constraint, alpha, critic_lr, gamma, device):
        state_dim = state_space.shape[0]
        action_dim = action_space.n
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda  # GAE 参数
        self.kl_constraint = kl_constraint  # KL 距离最大限制
        self.alpha = alpha  # 线性搜索的衰减系数
        self.device = device

    def take_action(self, state) -> int:
        state = torch.as_tensor(np.array([state], dtype=np.float32),
                                device=self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        return int(action_dist.sample().item())

    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        """计算黑塞矩阵（KL 的二阶导）与向量的乘积。"""
        new_action_dists = torch.distributions.Categorical(self.actor(states))
        kl = torch.mean(torch.distributions.kl.kl_divergence(
            old_action_dists, new_action_dists))
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(),
                                      create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product,
                                    self.actor.parameters())
        return torch.cat([grad.contiguous().view(-1) for grad in grad2])

    def conjugate_gradient(self, grad, states, old_action_dists):
        """共轭梯度法求解 H x = g。"""
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        for _ in range(10):
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)
            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs,
                              actor):
        log_probs = torch.log(actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def line_search(self, states, actions, advantage, old_log_probs,
                    old_action_dists, max_vec):
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantage,
                                             old_log_probs, self.actor)
        for i in range(15):
            coef = self.alpha ** i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(
                new_para, new_actor.parameters())
            new_action_dists = torch.distributions.Categorical(new_actor(states))
            kl_div = torch.mean(torch.distributions.kl.kl_divergence(
                old_action_dists, new_action_dists))
            new_obj = self.compute_surrogate_obj(states, actions, advantage,
                                                 old_log_probs, new_actor)
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return new_para
        return old_para

    def policy_learn(self, states, actions, old_action_dists, old_log_probs,
                     advantage):
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage,
                                                   old_log_probs, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        descent_direction = self.conjugate_gradient(obj_grad, states,
                                                    old_action_dists)
        Hd = self.hessian_matrix_vector_product(states, old_action_dists,
                                                descent_direction)
        max_coef = torch.sqrt(2 * self.kl_constraint /
                              (torch.dot(descent_direction, Hd) + 1e-8))
        new_para = self.line_search(states, actions, advantage, old_log_probs,
                                    old_action_dists,
                                    descent_direction * max_coef)
        torch.nn.utils.convert_parameters.vector_to_parameters(
            new_para, self.actor.parameters())

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
        old_action_dists = torch.distributions.Categorical(
            self.actor(states).detach())
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()  # 更新价值函数
        self.policy_learn(states, actions, old_action_dists, old_log_probs,
                          advantage)  # 更新策略函数


class PolicyNetContinuous(torch.nn.Module):
    """连续动作策略网络：输出高斯分布的均值与标准差。"""

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


class TRPOContinuous(TRPO):
    """连续动作的 TRPO 算法。"""

    def __init__(self, hidden_dim, state_space, action_space, lmbda,
                 kl_constraint, alpha, critic_lr, gamma, device):
        state_dim = state_space.shape[0]
        action_dim = action_space.shape[0]
        self.actor = PolicyNetContinuous(state_dim, hidden_dim,
                                         action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.kl_constraint = kl_constraint
        self.alpha = alpha
        self.device = device

    def take_action(self, state):
        state = torch.as_tensor(np.array([state], dtype=np.float32),
                                device=self.device)
        mu, std = self.actor(state)
        action_dist = torch.distributions.Normal(mu, std)
        action = action_dist.sample()
        return [action.item()]

    def hessian_matrix_vector_product(self, states, old_action_dists, vector,
                                      damping: float = 0.1):
        mu, std = self.actor(states)
        new_action_dists = torch.distributions.Normal(mu, std)
        kl = torch.mean(torch.distributions.kl.kl_divergence(
            old_action_dists, new_action_dists))
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(),
                                      create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product,
                                    self.actor.parameters())
        grad2_vector = torch.cat(
            [grad.contiguous().view(-1) for grad in grad2])
        return grad2_vector + damping * vector

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs,
                              actor):
        mu, std = actor(states)
        action_dists = torch.distributions.Normal(mu, std)
        log_probs = action_dists.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def line_search(self, states, actions, advantage, old_log_probs,
                    old_action_dists, max_vec):
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantage,
                                             old_log_probs, self.actor)
        for i in range(15):
            coef = self.alpha ** i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(
                new_para, new_actor.parameters())
            mu, std = new_actor(states)
            new_action_dists = torch.distributions.Normal(mu, std)
            kl_div = torch.mean(torch.distributions.kl.kl_divergence(
                old_action_dists, new_action_dists))
            new_obj = self.compute_surrogate_obj(states, actions, advantage,
                                                 old_log_probs, new_actor)
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return new_para
        return old_para

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
        rewards = (rewards + 8.0) / 8.0  # 对奖励做缩放，方便训练
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                      td_delta.cpu()).to(self.device)
        mu, std = self.actor(states)
        old_action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = old_action_dists.log_prob(actions)
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.policy_learn(states, actions, old_action_dists, old_log_probs,
                          advantage)


def main() -> None:
    parser = argparse.ArgumentParser(description="第 11 章 TRPO 算法")
    parser.add_argument("--env", default="cartpole",
                        choices=["cartpole", "pendulum"])
    parser.add_argument("--episodes", type=int, default=None,
                        help="默认 CartPole 500、Pendulum 2000（书中设置）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    hidden_dim = 128
    device = get_device(args.cuda)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.env == "cartpole":
        env_name = "CartPole-v1"
        env = gym.make(env_name)
        episodes = args.episodes if args.episodes is not None else 500
        agent = TRPO(hidden_dim, env.observation_space, env.action_space,
                     lmbda=0.95, kl_constraint=0.0005, alpha=0.5,
                     critic_lr=1e-2, gamma=0.98, device=device)
        tag = "ch11_trpo_cartpole.png"
    else:
        env_name = "Pendulum-v1"
        env = gym.make(env_name)
        episodes = args.episodes if args.episodes is not None else 2000
        agent = TRPOContinuous(hidden_dim, env.observation_space,
                               env.action_space, lmbda=0.9,
                               kl_constraint=0.00005, alpha=0.5,
                               critic_lr=1e-2, gamma=0.9, device=device)
        tag = "ch11_trpo_pendulum.png"

    env.reset(seed=args.seed)
    return_list = train_on_policy_agent(env, agent, episodes, seed=args.seed)
    print(f"最后 20 回合平均回报: {np.mean(return_list[-20:]):.3f}")
    path = plot_returns(return_list, f"TRPO on {env_name}", tag)
    print(f"图像已保存: {path}")
    env.close()


if __name__ == "__main__":
    main()
