"""绘图小工具：统一使用无界面后端，并把图片保存到 outputs/figures。"""

from __future__ import annotations

import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "outputs" / "figures"

# 指定一个可写的 matplotlib 缓存目录，避免在只读 HOME 下报警告
_MPL_CACHE = ROOT / "outputs" / ".mplconfig"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")  # 服务器/无显示环境下也能出图
import matplotlib.pyplot as plt
import numpy as np


def moving_average(values, window: int = 10):
    """滑动平均，方便观察收敛趋势。"""
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def save_fig(fig, name: str) -> pathlib.Path:
    """保存并关闭图像，返回文件路径。"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_returns(return_list, title: str, filename: str, window: int = 9,
                 ylabel: str = "Returns"):
    """画回报曲线：原始曲线 + 滑动平均，保存并返回路径。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(return_list)), return_list, alpha=0.35, label="return")
    mv = moving_average(return_list, window)
    ax.plot(range(len(return_list) - len(mv), len(return_list)), mv,
            linewidth=2, label=f"moving average ({window})")
    ax.set_xlabel("Episodes")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return save_fig(fig, filename)
