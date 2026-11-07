"""《动手学强化学习》(https://hrl.boyuai.com) 第 1~6 章示例代码的 gymnasium 版本。

模块说明：
  * ``envs``     —— 书中 CliffWalkingEnv 的 gymnasium.Env 实现与注册
  * ``plotting`` —— 无界面绘图与图像保存
"""

from .envs import (  # noqa: F401
    CliffWalkingBookEnv,
    make_env,
    grid_shape,
    register_envs,
)

__all__ = ["CliffWalkingBookEnv", "make_env", "grid_shape", "register_envs"]
