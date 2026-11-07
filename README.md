# 动手学强化学习 · 第 1~21 章示例代码（gymnasium 版）

本目录把《动手学强化学习》（<https://hrl.boyuai.com/chapter/intro>）全书
21 章的示例代码整理成了可直接运行的 `.py` 文件，并把书中基于旧版 `gym`
的写法迁移到 **gymnasium** 接口（`reset -> (obs, info)`、
`step -> (obs, reward, terminated, truncated, info)`）。

- 第 1~6 章：表格型方法，只依赖 numpy / matplotlib。
- 第 7~21 章：深度强化学习，需要 PyTorch。

## 章节与文件对应

| 章 | 文件 | 内容 |
| --- | --- | --- |
| 1 初探强化学习 | `chapter01_intro.py` | 书中无代码；补充 gymnasium 交互循环与折扣回报演示 |
| 2 多臂老虎机 | `chapter02_bandit.py` | 伯努利老虎机、ε-贪婪、ε 衰减、UCB、汤普森采样 |
| 3 马尔可夫决策过程 | `chapter03_mdp.py` | 回报、贝尔曼方程解析解、轨迹采样、蒙特卡洛、占用度量 |
| 4 动态规划 | `chapter04_dp.py` | 策略迭代、价值迭代（悬崖漫步 + FrozenLake-v1） |
| 5 时序差分 | `chapter05_td.py` | Sarsa、n 步 Sarsa、Q-learning |
| 6 Dyna-Q | `chapter06_dynaq.py` | Dyna-Q，对比 Q-planning 0/2/20 步 |
| 7 DQN | `chapter07_dqn.py` | 经验回放 + 目标网络，CartPole-v1 |
| 8 DQN 改进 | `chapter08_dqn_improved.py` | Double DQN / Dueling DQN，Pendulum 离散化 |
| 9 策略梯度 | `chapter09_reinforce.py` | REINFORCE，CartPole-v1 |
| 10 Actor-Critic | `chapter10_actor_critic.py` | Actor-Critic（TD 误差），CartPole-v1 |
| 11 TRPO | `chapter11_trpo.py` | 共轭梯度 + 线性搜索，离散 CartPole / 连续 Pendulum |
| 12 PPO | `chapter12_ppo.py` | 截断目标 PPO，离散 CartPole / 连续 Pendulum |
| 13 DDPG | `chapter13_ddpg.py` | 确定性策略梯度 + 软更新，Pendulum |
| 14 SAC | `chapter14_sac.py` | 双 Q + 最大熵 + 自动温度，Pendulum / CartPole |
| 15 模仿学习 | `chapter15_imitation.py` | PPO 专家 → 行为克隆（BC）→ GAIL |
| 16 模型预测控制 | `chapter16_mpc.py` | PETS：集成环境模型 + CEM 规划（模型训练轮数默认调大） |
| 17 基于模型的策略优化 | `chapter17_mbpo.py` | MBPO：集成动力学模型 + SAC |
| 18 离线强化学习 | `chapter18_offline_rl.py` | 先在线收集数据，再用 CQL 离线训练 |
| 19 目标导向的强化学习 | `chapter19_goal_rl.py` | 自写 WorldEnv + DDPG + HER |
| 20 多智能体入门 | `chapter20_marl_basic.py` | IPPO（参数共享 PPO）2v2 Combat |
| 21 多智能体进阶 | `chapter21_marl_advanced.py` | MADDPG，1 捣乱者 + 2 协作智能体 |

```
.
├── chapter01_intro.py ~ chapter21_marl_advanced.py   # 每章一个脚本，可独立运行
├── hrl/                                              # 公用代码
│   ├── envs.py      # 书中 CliffWalkingEnv 的 gymnasium.Env 实现，注册为 CliffWalkingBook-v0
│   ├── rl_utils.py  # 书中 rl_utils 的 gymnasium 版（回放池、训练循环、GAE 等）
│   └── plotting.py  # 无界面绘图（图片保存到 outputs/figures）
├── outputs/figures/  # 运行后生成的学习曲线
├── requirements.txt
└── run_all.sh        # 21 章的快速冒烟测试
```

## 环境与运行

第 1~6 章：

```bash
pip install numpy matplotlib
python chapter05_td.py
```

第 7~21 章需要 PyTorch（CPU 即可）。以本机为例：

```bash
/home/zhy-2204/miniconda3/envs/env_isaaclab/bin/python chapter07_dqn.py
```

每个脚本都支持 `--help`，并且都有 `--episodes`（部分脚本还有 `--epochs` 等）
用来把书中较长的训练缩短，便于快速验证：

```bash
python chapter07_dqn.py --episodes 150
python chapter11_trpo.py --env pendulum --episodes 200     # 书中默认 2000
python chapter17_mbpo.py --episodes 2 --epochs 1
python chapter18_offline_rl.py --episodes 5 --collect-episodes 5 --grad-steps 50
```

一次性跑完 21 章（快速参数，几分钟）：

```bash
PYTHON=/home/zhy-2204/miniconda3/envs/env_isaaclab/bin/python bash run_all.sh
```

依赖：Python ≥ 3.10、gymnasium ≥ 1.0、numpy、matplotlib、tqdm、torch。
绘图使用 `Agg` 后端，只保存图片、不弹窗，适合服务器/无显示环境。

## 相对书中代码的主要改动

1. **环境接口**：`gym` → `gymnasium`；`reset()` 返回 `(obs, info)`，
   `step()` 返回五元组，循环用 `terminated or truncated` 判断回合结束，
   `env.seed(0)` 全部改为 `env.reset(seed=0)`。
2. **环境获取**：全部通过 `gymnasium.make(...)` 创建。
   书中自写的 `CliffWalkingEnv` 改写为 `gymnasium.Env` 子类（`hrl/envs.py`）
   并注册为 `CliffWalkingBook-v0`；`CartPole-v0` → `CartPole-v1`、
   `Pendulum-v0` → `Pendulum-v1`、`FrozenLake-v0` → `FrozenLake-v1`。
3. **rl_utils**：第 7 章之后的示例原本 `import rl_utils`，这里用
   `hrl/rl_utils.py` 代替，保留 `ReplayBuffer`、`moving_average`、
   `train_on_policy_agent`、`train_off_policy_agent`、`compute_advantage`
   等原函数名与签名，只把内部循环改写为 gymnasium 五元组接口。
4. **绘图**：`plt.show()` 改为保存到 `outputs/figures/*.png`，
   并统一叠加滑动平均曲线。
5. **CPU 单线程**：这些示例网络很小，PyTorch 在多核 CPU 上开多线程反而慢数倍，
   因此 `hrl/rl_utils.py` 默认 `torch.set_num_threads(1)`
   （可用环境变量 `HRL_TORCH_THREADS` 覆盖）。
6. **两处外部环境替代**（第 20、21 章）：书中依赖
   `ma-gym` 与 `multiagent-particle-envs` 两个需要联网 clone、且锁死旧版 gym 的仓库，
   本目录内置了等价的简化环境（接口与算法保持一致，细节见下文）。

### 与 gymnasium 内置 CliffWalking-v1 的差异

书中悬崖漫步与 `gymnasium.make("CliffWalking-v1")` 并不完全一致：

| 项目 | 书中版本（`CliffWalkingBook-v0`） | gymnasium `CliffWalking-v1` |
| --- | --- | --- |
| 动作顺序 | 0 上、1 下、2 左、3 右 | 0 上、1 右、2 下、3 左 |
| 掉下悬崖 | `reward=-100`，**回合结束** | `reward=-100`，**回到起点、回合继续** |
| 到达终点 | `reward=-1`，回合结束 | `reward=-1`，回合结束 |

为了让结果与书中一致，第 4~6 章默认使用 `CliffWalkingBook-v0`；
加 `--env gym` 可以切换到内置的 `CliffWalking-v1` 做对比。

### 第 20、21 章的环境替代

- **第 20 章 Combat**：书中用 `ma-gym` 的 15×15、2v2 Combat。这里内置
  `CombatLite`：同样的 15×15 格子、2 个智能体对 2 个敌人、生命值 3、
  5 个动作（上/下/左/右/攻击），观测含自己、队友和两个敌人的相对位置与血量；
  击败全部敌人即获胜。训练流程（参数共享 PPO、赢一局 +100 奖励）与书中一致。
- **第 21 章 simple_adversary**：书中用 `multiagent-particle-envs`。这里内置
  `SimpleAdversaryLite`：1 个捣乱者 + 2 个协作智能体 + 1 个随机地标，
  离散 5 动作、每步移动 0.1，协作智能体能看到地标、捣乱者看不到；
  奖励分别为「到地标距离」的负值。MADDPG 的集中式 Critic、Gumbel-Softmax
  离散动作等实现与书中一致。

## 结果对照

第 1~6 章的输出与书中完全一致（可用来校验代码正确性）：

- 第 2 章：ε-贪婪累积懊悔 `25.5266`、ε 衰减 `10.1143`、UCB `70.4528`、
  汤普森采样 `57.1916`（沿用书中的 `np.random.seed`）。
- 第 3 章：MRP 解析解 `[-2.0195, -2.2145, 1.1614, 10.5381, 3.5873, 0]`，
  MDP 解析解 `[-1.2256, -1.6767, 0.5189, 6.0756, 0]`，
  蒙特卡洛估值与占用度量 `0.1126 / 0.2320`。
- 第 4 章：悬崖漫步策略迭代评估轮数 `60/72/44/12/1`、价值迭代 `14` 轮，
  状态价值 `-7.712, -7.458, ...`；冰湖策略迭代 `25/58` 轮、价值迭代 `60` 轮，
  状态价值 `0.069, 0.061, ...`。
- 第 6 章：planning 步数越多收敛越快（最后 20 回合平均回报约
  `-26.6 → -13.1 → -13.2`）。

第 7~21 章是随机性很强的深度 RL，曲线不会与书中逐点相同，但趋势一致
（以下为本机 CPU 实测）：

| 章 | 实测（缩短训练后） |
| --- | --- |
| 7 DQN | CartPole 150 回合后平均回报约 267 |
| 8 Double DQN | Pendulum 200 回合后约 -246（初始 -1300） |
| 9 REINFORCE | CartPole 200 回合后明显上升 |
| 10 Actor-Critic | CartPole 300 回合后约 117 |
| 11/12 TRPO/PPO | CartPole 快速收敛；Pendulum 连续动作版可正常训练 |
| 13 DDPG | Pendulum 回报从 -1400 上升到 -200 左右（书中默认 200 回合） |
| 14 SAC | CartPole 300 回合后约 316（前 150 回合基本不学，属正常） |
| 15 BC/GAIL | 只用 1 条专家轨迹即可让 GAIL 的回报明显高于随机 |
| 16 PETS | 10 回合后平均回报约 -700（首回合约 -1800） |
| 17 MBPO | 20 回合后最后 5 回合平均约 -191（末回合 -3） |
| 18 CQL | 离线数据上训练后可在真实环境评估 |
| 19 HER | 300 回合后成功率可达 100%（`--episodes 300 --skip-baseline`） |
| 20 IPPO | 9×9 小地图 100 回合胜率约 0.45，并随训练继续上升 |
| 21 MADDPG | 3 个智能体的回报随训练上升 |

> 提示：第 16、17 章的 `--epochs` 默认分别是 100、50（书中都是 20）——
> 环境模型训练轮数太少会明显拖慢效果，调大后 CEM 规划 / MBPO 的提升才看得出来。
>
> 提示：gymnasium 的 `CartPole-v1` 单回合上限是 500 步（书中 v0 是 200 步），
> 因此回报数值不能与书中直接比较；Pendulum 的奖励范围是 -16~0。
