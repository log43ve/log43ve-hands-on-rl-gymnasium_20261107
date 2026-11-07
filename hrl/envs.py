"""环境与通用工具。

《动手学强化学习》书中第 4~6 章使用的是一个自写的 CliffWalkingEnv（接口是
``reset() -> state``、``step(a) -> (state, reward, done)``，属于旧版 gym 风格）。
为了让示例代码可以直接在 **gymnasium** 下运行，这里把它改写成一个标准的
``gymnasium.Env``：

* ``reset(seed=None)`` 返回 ``(obs, info)``
* ``step(a)`` 返回 ``(obs, reward, terminated, truncated, info)``
* 使用 ``gymnasium.spaces.Discrete`` 描述状态/动作空间
* 保留 ``P[s][a] = [(prob, next_state, reward, terminated)]`` 转移模型，
  供第 4 章动态规划直接使用（与 ``FrozenLake-v1`` 的 ``unwrapped.P`` 格式一致）

除此之外还注册了 ``CliffWalkingBook-v0``，于是可以像内置环境一样通过
``gymnasium.make("CliffWalkingBook-v0")`` 创建它。
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# 书中动作编号：0 上、1 下、2 左、3 右，(dx, dy) 以左上角为原点、y 轴向下
CHANGE = ((0, -1), (0, 1), (-1, 0), (1, 0))
ACTION_MEANING = ("^", "v", "<", ">")

CLIFF_ID = "CliffWalkingBook-v0"


class CliffWalkingBookEnv(gym.Env):
    """悬崖漫步（cliff walking），动力学与书中第 4~6 章完全一致。

    与 gymnasium 内置 ``CliffWalking-v1`` 的差别：

    ==============  =========================  ============================
    项目             本环境（书中版本）           gymnasium CliffWalking-v1
    ==============  =========================  ============================
    动作顺序         0上 1下 2左 3右            0上 1右 2下 3左
    掉下悬崖         回合结束(reward=-100)       回到起点，回合继续
    到达终点         reward=-1，回合结束         reward=-1，回合结束
    ==============  =========================  ============================
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(self, ncol: int = 12, nrow: int = 4, render_mode: str | None = None):
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.ncol = int(ncol)
        self.nrow = int(nrow)
        self.nS = self.nrow * self.ncol
        self.nA = 4
        self.render_mode = render_mode

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)

        self.P = self._build_P()

        self.x = 0
        self.y = self.nrow - 1

    # ------------------------------------------------------------------ 模型
    def _build_P(self):
        """构造 P[s][a] = [(prob, next_state, reward, terminated)]。"""
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        for i in range(self.nrow):
            for j in range(self.ncol):
                s = i * self.ncol + j
                # 悬崖或终点：无法继续交互，任何动作奖励都为 0
                if i == self.nrow - 1 and j > 0:
                    for a in range(self.nA):
                        P[s][a] = [(1.0, s, 0.0, True)]
                    continue
                for a in range(self.nA):
                    next_x = min(self.ncol - 1, max(0, j + CHANGE[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + CHANGE[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward, done = -1.0, False
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # 下一个位置在悬崖
                            reward = -100.0
                    P[s][a] = [(1.0, next_state, reward, done)]
        return P

    # ------------------------------------------------------------------ 交互
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x, {"prob": 1.0}

    def step(self, action: int):
        state = self.y * self.ncol + self.x
        prob, next_state, reward, terminated = self.P[state][int(action)][0]
        self.y, self.x = divmod(next_state, self.ncol)
        return int(next_state), float(reward), bool(terminated), False, {"prob": prob}

    # ------------------------------------------------------------------ 渲染
    def render(self):
        if self.render_mode is None:
            gym.logger.warn("需要设置 render_mode='ansi' 才能调用 render()")
            return None
        line = []
        for i in range(self.nrow):
            for j in range(self.ncol):
                s = i * self.ncol + j
                if (self.x, self.y) == (j, i):
                    cell = " x "
                elif i == self.nrow - 1 and j == self.ncol - 1:
                    cell = " T "
                elif i == self.nrow - 1 and j > 0:
                    cell = " C "
                else:
                    cell = " o "
                line.append(cell)
            line.append("\n")
        text = "".join(line)
        if self.render_mode == "human":
            print(text)
        return text


def register_envs() -> None:
    """把书中的环境注册进 gymnasium，使其可以 gymnasium.make(...) 创建。"""
    if CLIFF_ID in gym.registry:
        return
    gym.register(id=CLIFF_ID, entry_point="hrl.envs:CliffWalkingBookEnv")


def make_env(name: str = "book", **kwargs) -> gym.Env:
    """按名字创建环境，统一走 gymnasium.make。

    name 可选：
      * ``book``   —— 书中动力学版本的悬崖漫步（CliffWalkingBook-v0）
      * ``gym``    —— gymnasium 内置的 CliffWalking-v1
      * ``frozen`` —— gymnasium 内置的 FrozenLake-v1
    """
    register_envs()
    name = name.lower()
    if name in ("book", "cliff", "cliffwalking-book"):
        return gym.make(CLIFF_ID, **kwargs)
    if name in ("gym", "builtin", "cliffwalking"):
        return gym.make("CliffWalking-v1", **kwargs)
    if name in ("frozen", "frozenlake"):
        return gym.make("FrozenLake-v1", **kwargs)
    raise ValueError(f"未知环境: {name}")


def grid_shape(env: gym.Env) -> tuple[int, int]:
    """返回环境的 (nrow, ncol)，兼容自定义环境与 gymnasium 内置环境。"""
    u = env.unwrapped
    if hasattr(u, "nrow") and hasattr(u, "ncol"):
        return int(u.nrow), int(u.ncol)
    if hasattr(u, "shape"):
        return int(u.shape[0]), int(u.shape[1])
    raise AttributeError("无法从环境中获取网格形状")
