"""第 16 章 模型预测控制：PETS（概率环境模型集成 + 轨迹采样）

对应书中示例：在 Pendulum 上先用随机策略采集数据，训练一个由若干全连接网络
组成的环境模型（输出状态增量与奖励的均值、方差），然后用 CEM（交叉熵方法）
在学到的“假环境”里滚动若干步来挑选动作，即模型预测控制（MPC）。

运行（需要 torch）：
    python chapter16_mpc.py
    python chapter16_mpc.py --episodes 2 --epochs 2 --n-sequence 10 --plan-horizon 10

默认超参数与书中一致：环境交互 10 回合、CEM 采样 50 条动作序列、
精英比例 0.2、预测步长 25、环境模型每回合最多训练 20 轮。
注意 gymnasium 中 Pendulum 只有 v1（v0 已移除）。

与书中代码的唯一依赖差异：书中用 ``scipy.stats.truncnorm`` 采样截断正态噪声，
这里用 numpy 的拒绝采样实现 ``truncated_normal_sample``（等价于 truncnorm(-2, 2)），
从而不依赖 scipy。
"""

from __future__ import annotations

import argparse
import collections
import itertools

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hrl.plotting import plot_returns
from hrl.rl_utils import get_device


def truncated_normal_sample(shape, bound: float = 2.0):
    """采样标准正态分布，并截断到 [-bound, bound]（等价于 truncnorm(-2, 2)）。"""
    samples = np.random.normal(size=shape)
    mask = np.abs(samples) > bound
    while np.any(mask):
        samples[mask] = np.random.normal(size=int(mask.sum()))
        mask = np.abs(samples) > bound
    return samples


class CEM:
    """交叉熵方法：迭代地“采样动作序列 -> 选精英 -> 更新均值方差”。"""

    def __init__(self, n_sequence, elite_ratio, fake_env, upper_bound,
                 lower_bound):
        self.n_sequence = n_sequence
        self.elite_ratio = elite_ratio
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.fake_env = fake_env

    def optimize(self, state, init_mean, init_var):
        mean, var = init_mean, init_var
        state = np.tile(state, (self.n_sequence, 1))

        for _ in range(5):
            lb_dist, ub_dist = mean - self.lower_bound, self.upper_bound - mean
            constrained_var = np.minimum(
                np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)),
                var)
            # 生成动作序列（截断正态噪声 * 标准差 + 均值）
            # 书中 Pendulum 的动作维度为 1，因此 mean 的长度就是预测步长
            noise = truncated_normal_sample((self.n_sequence, mean.shape[0]))
            action_sequences = noise * np.sqrt(constrained_var) + mean
            # 计算每条动作序列的累积奖励
            returns = self.fake_env.propagate(state, action_sequences)[:, 0]
            # 选取累积奖励最高的若干条动作序列
            elites = action_sequences[np.argsort(returns)][-int(
                self.elite_ratio * self.n_sequence):]
            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)
            # 更新动作序列分布
            mean = 0.1 * mean + 0.9 * new_mean
            var = 0.1 * var + 0.9 * new_var

        return mean


class Swish(nn.Module):
    ''' Swish 激活函数 '''

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def truncated_normal_init_(t, mean: float = 0.0, std: float = 0.01):
    """截断正态初始化：把偏离均值超过两倍标准差的权重重新采样。"""
    torch.nn.init.normal_(t, mean=mean, std=std)
    while True:
        cond = (t < mean - 2 * std) | (t > mean + 2 * std)
        if not torch.sum(cond):
            break
        with torch.no_grad():
            t[cond] = torch.normal(mean, std, size=(int(cond.sum()),),
                                   device=t.device)
    return t


def init_weights(m):
    ''' 初始化模型权重 '''
    if isinstance(m, FCLayer):
        truncated_normal_init_(m.weight, std=1 / (2 * np.sqrt(m._input_dim)))
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Linear):
        truncated_normal_init_(m.weight, std=1 / (2 * np.sqrt(m.in_features)))
        m.bias.data.fill_(0.0)


class FCLayer(nn.Module):
    ''' 集成之后的全连接层（一次算完 ensemble_size 个网络）'''

    def __init__(self, input_dim, output_dim, ensemble_size, activation,
                 device=None):
        super().__init__()
        self._input_dim, self._output_dim = input_dim, output_dim
        self.device = device if device is not None else torch.device("cpu")
        self.weight = nn.Parameter(
            torch.Tensor(ensemble_size, input_dim, output_dim).to(self.device))
        self._activation = activation
        self.bias = nn.Parameter(
            torch.Tensor(ensemble_size, output_dim).to(self.device))

    def forward(self, x):
        return self._activation(
            torch.add(torch.bmm(x, self.weight), self.bias[:, None, :]))


class EnsembleModel(nn.Module):
    ''' 环境模型集成：一次输出 ensemble_size 个网络的 (均值, 方差) '''

    def __init__(self, state_dim, action_dim, ensemble_size=5,
                 learning_rate=1e-3, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        # 输出包括均值和方差，因此是状态与奖励维度之和的两倍
        self._output_dim = (state_dim + 1) * 2
        self._max_logvar = nn.Parameter(
            (torch.ones((1, self._output_dim // 2)).float() / 2).to(
                self.device), requires_grad=False)
        self._min_logvar = nn.Parameter(
            (-torch.ones((1, self._output_dim // 2)).float() * 10).to(
                self.device), requires_grad=False)

        self.layer1 = FCLayer(state_dim + action_dim, 200, ensemble_size,
                              Swish(), self.device)
        self.layer2 = FCLayer(200, 200, ensemble_size, Swish(), self.device)
        self.layer3 = FCLayer(200, 200, ensemble_size, Swish(), self.device)
        self.layer4 = FCLayer(200, 200, ensemble_size, Swish(), self.device)
        self.layer5 = FCLayer(200, self._output_dim, ensemble_size,
                              nn.Identity(), self.device)
        self.apply(init_weights)  # 初始化环境模型中的参数
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, return_log_var=False):
        ret = self.layer5(self.layer4(self.layer3(self.layer2(
            self.layer1(x)))))
        mean = ret[:, :, :self._output_dim // 2]
        # 在 PETS 算法中，将方差控制在最小值和最大值之间
        logvar = self._max_logvar - F.softplus(
            self._max_logvar - ret[:, :, self._output_dim // 2:])
        logvar = self._min_logvar + F.softplus(logvar - self._min_logvar)
        return mean, logvar if return_log_var else torch.exp(logvar)

    def loss(self, mean, logvar, labels, use_var_loss=True):
        inverse_var = torch.exp(-logvar)
        if use_var_loss:
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) *
                                             inverse_var,
                                             dim=-1),
                                  dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train(self, loss):
        # 书中重载了 nn.Module.train：这里只用它按损失更新一次参数
        self.optimizer.zero_grad()
        loss += 0.01 * torch.sum(self._max_logvar) - 0.01 * torch.sum(
            self._min_logvar)
        loss.backward()
        self.optimizer.step()


class EnsembleDynamicsModel:
    ''' 环境模型集成，加入精细化的训练（早停 + 保存最优快照）'''

    def __init__(self, state_dim, action_dim, num_network=5, device=None):
        self._num_network = num_network
        self._state_dim, self._action_dim = state_dim, action_dim
        self.device = device if device is not None else torch.device("cpu")
        self.model = EnsembleModel(state_dim, action_dim,
                                   ensemble_size=num_network,
                                   device=self.device)
        self._epoch_since_last_update = 0

    def train(self,
              inputs,
              labels,
              batch_size=64,
              holdout_ratio=0.1,
              max_iter=20):
        # 设置训练集与验证集
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]
        num_holdout = int(inputs.shape[0] * holdout_ratio)
        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = (inputs[:num_holdout],
                                          labels[:num_holdout])
        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(
            self.device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(
            self.device)
        holdout_inputs = holdout_inputs[None, :, :].repeat(
            [self._num_network, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat(
            [self._num_network, 1, 1])

        # 保留最好的结果
        self._snapshots = {i: (None, 1e10) for i in range(self._num_network)}

        for epoch in itertools.count():
            # 定义每一个网络的训练数据
            train_index = np.vstack([
                np.random.permutation(train_inputs.shape[0])
                for _ in range(self._num_network)
            ])
            # 所有真实数据都用来训练
            for batch_start_pos in range(0, train_inputs.shape[0], batch_size):
                batch_index = train_index[:, batch_start_pos:batch_start_pos +
                                          batch_size]
                train_input = torch.from_numpy(
                    train_inputs[batch_index]).float().to(self.device)
                train_label = torch.from_numpy(
                    train_labels[batch_index]).float().to(self.device)

                mean, logvar = self.model(train_input, return_log_var=True)
                loss, _ = self.model.loss(mean, logvar, train_label)
                self.model.train(loss)

            with torch.no_grad():
                mean, logvar = self.model(holdout_inputs, return_log_var=True)
                _, holdout_losses = self.model.loss(mean,
                                                    logvar,
                                                    holdout_labels,
                                                    use_var_loss=False)
                holdout_losses = holdout_losses.cpu()
                break_condition = self._save_best(epoch, holdout_losses)
                if break_condition or epoch > max_iter:  # 结束训练
                    break

    def _save_best(self, epoch, losses, threshold=0.1):
        updated = False
        for i in range(len(losses)):
            current = losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > threshold:
                self._snapshots[i] = (epoch, current)
                updated = True
        self._epoch_since_last_update = 0 if updated else self._epoch_since_last_update + 1
        return self._epoch_since_last_update > 5

    def predict(self, inputs, batch_size=64):
        mean, var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = torch.from_numpy(
                inputs[i:min(i + batch_size, inputs.shape[0])]).float().to(
                    self.device)
            cur_mean, cur_var = self.model(input[None, :, :].repeat(
                [self._num_network, 1, 1]),
                                           return_log_var=False)
            mean.append(cur_mean.detach().cpu().numpy())
            var.append(cur_var.detach().cpu().numpy())
        return np.hstack(mean), np.hstack(var)


class FakeEnv:
    ''' 用学到的环境模型“假装”出来的环境 '''

    def __init__(self, model):
        self.model = model

    def step(self, obs, act):
        obs = np.asarray(obs, dtype=np.float32)
        act = np.asarray(act, dtype=np.float32)
        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        ensemble_model_means[:, :, 1:] += obs  # 模型预测的是状态增量
        ensemble_model_stds = np.sqrt(ensemble_model_vars)
        # 从“均值 + 标准差 * 噪声”中采样，等价于从各网络的高斯分布中采样
        ensemble_samples = ensemble_model_means + np.random.normal(
            size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape
        models_to_use = np.random.choice(
            [i for i in range(self.model._num_network)], size=batch_size)
        batch_inds = np.arange(0, batch_size)
        samples = ensemble_samples[models_to_use, batch_inds]
        rewards, next_obs = samples[:, :1], samples[:, 1:]
        return rewards, next_obs

    def propagate(self, obs, actions):
        ''' 在模型里滚动预测，返回每条动作序列的累积奖励 '''
        with torch.no_grad():
            obs = np.copy(obs).astype(np.float32)
            total_reward = np.zeros((obs.shape[0], 1), dtype=np.float32)
            actions = np.asarray(actions, dtype=np.float32)
            for i in range(actions.shape[1]):
                action = actions[:, i:i + 1]
                rewards, next_obs = self.step(obs, action)
                total_reward += rewards
                obs = np.asarray(next_obs, dtype=np.float32)
            return total_reward


class ReplayBuffer:
    """经验回放池（这里需要取出全部样本用于训练环境模型）。"""

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def size(self):
        return len(self.buffer)

    def return_all_samples(self):
        all_transitions = list(self.buffer)
        state, action, reward, next_state, done = zip(*all_transitions)
        return np.array(state), action, reward, np.array(next_state), done


class PETS:
    ''' PETS 算法（Probabilistic Ensembles with Trajectory Sampling）'''

    def __init__(self, env, replay_buffer, n_sequence, elite_ratio,
                 plan_horizon, num_episodes, device, max_iter=20):
        self._env = env
        self._env_pool = replay_buffer

        obs_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]
        self._model = EnsembleDynamicsModel(obs_dim, self._action_dim,
                                            device=device)
        self._fake_env = FakeEnv(self._model)
        self.upper_bound = env.action_space.high[0]
        self.lower_bound = env.action_space.low[0]

        self._cem = CEM(n_sequence, elite_ratio, self._fake_env,
                        self.upper_bound, self.lower_bound)
        self.plan_horizon = plan_horizon
        self.num_episodes = num_episodes
        self._max_iter = max_iter

    def train_model(self):
        env_samples = self._env_pool.return_all_samples()
        obs = env_samples[0]
        actions = np.array(env_samples[1])
        rewards = np.array(env_samples[2]).reshape(-1, 1)
        next_obs = env_samples[3]
        inputs = np.concatenate((obs, actions), axis=-1)
        labels = np.concatenate((rewards, next_obs - obs), axis=-1)
        self._model.train(inputs, labels, max_iter=self._max_iter)

    def mpc(self):
        mean = np.tile((self.upper_bound + self.lower_bound) / 2.0,
                       self.plan_horizon)
        var = np.tile(
            np.square(self.upper_bound - self.lower_bound) / 16,
            self.plan_horizon)
        obs, info = self._env.reset()
        done, episode_return = False, 0.0
        while not done:
            actions = self._cem.optimize(obs, mean, var)
            # 选取第一个动作
            action = np.asarray(actions[:self._action_dim], dtype=np.float32)
            next_obs, reward, terminated, truncated, info = self._env.step(
                action)
            done = terminated or truncated
            self._env_pool.add(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_return += reward
            # 把动作序列往后挪一位，作为下一步 CEM 的初始均值（热启动）
            mean = np.concatenate([
                np.copy(actions)[self._action_dim:],
                np.zeros(self._action_dim)
            ])
        return episode_return

    def explore(self):
        obs, info = self._env.reset()
        done, episode_return = False, 0.0
        while not done:
            action = self._env.action_space.sample()
            next_obs, reward, terminated, truncated, info = self._env.step(
                action)
            done = terminated or truncated
            self._env_pool.add(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_return += reward
        return episode_return

    def train(self):
        return_list = []
        # 先进行随机策略的探索来收集一条序列的数据
        explore_return = self.explore()
        print('episode: 1, return: %.1f' % explore_return)
        return_list.append(explore_return)

        for i_episode in range(self.num_episodes - 1):
            self.train_model()
            episode_return = self.mpc()
            return_list.append(episode_return)
            print('episode: %d, return: %.1f' % (i_episode + 2, episode_return))
        return return_list


def main() -> None:
    parser = argparse.ArgumentParser(description="第 16 章 模型预测控制（PETS）")
    parser.add_argument("--episodes", type=int, default=10,
                        help="与环境交互的回合数（书中 10）")
    parser.add_argument("--epochs", type=int, default=100,
                        help="每回合训练环境模型的最大迭代轮数（书中 20；"
                             "轮数太少模型不准，CEM 规划效果会很差）")
    parser.add_argument("--n-sequence", type=int, default=50,
                        help="CEM 采样的动作序列条数（书中 50）")
    parser.add_argument("--elite-ratio", type=float, default=0.2,
                        help="CEM 精英比例（书中 0.2）")
    parser.add_argument("--plan-horizon", type=int, default=25,
                        help="MPC 的预测步长（书中 25）")
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true", help="有 GPU 时使用 GPU")
    args = parser.parse_args()

    device = get_device(args.cuda)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_name = "Pendulum-v1"
    env = gym.make(env_name)
    env.reset(seed=args.seed)

    replay_buffer = ReplayBuffer(args.buffer_size)
    pets = PETS(env, replay_buffer, args.n_sequence, args.elite_ratio,
                args.plan_horizon, args.episodes, device,
                max_iter=args.epochs)
    return_list = pets.train()

    print(f"最后 20 回合平均回报: {np.mean(return_list[-20:]):.3f}")
    path = plot_returns(return_list, f"PETS on {env_name}", "ch16_pets.png")
    print(f"图像已保存: {path}")
    env.close()


if __name__ == "__main__":
    main()
