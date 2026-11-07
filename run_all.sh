#!/usr/bin/env bash
# 第 1~21 章的快速冒烟测试：
#   bash run_all.sh                # 使用当前 python
#   PYTHON=/path/to/python bash run_all.sh
#
# 这里为了快速跑完，参数都比书中的默认值小；想复现书中的曲线，
# 直接运行对应脚本、使用它自己的默认参数即可。
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

run() {
    echo
    echo "=================== $* ==================="
    "$PY" "$@"
}

# ---- 基础篇（第 1~6 章，只需 numpy/matplotlib）----
run chapter01_intro.py
run chapter02_bandit.py --steps 2000
run chapter03_mdp.py
run chapter04_dp.py
run chapter05_td.py --episodes 200
run chapter06_dynaq.py --episodes 100

# ---- 进阶篇（第 7~14 章，需要 torch）----
run chapter07_dqn.py --episodes 100
run chapter08_dqn_improved.py --episodes 30
run chapter09_reinforce.py --episodes 100
run chapter10_actor_critic.py --episodes 150
run chapter11_trpo.py --env cartpole --episodes 60
run chapter12_ppo.py --env cartpole --episodes 60
run chapter13_ddpg.py --episodes 100
run chapter14_sac.py --env cartpole --episodes 250

# ---- 前沿篇（第 15~21 章，需要 torch）----
run chapter15_imitation.py --episodes 30 --ppo-episodes 20 --bc-iterations 200
run chapter16_mpc.py --episodes 2 --epochs 1
run chapter17_mbpo.py --episodes 2 --epochs 1
run chapter18_offline_rl.py --episodes 5 --collect-episodes 5 --grad-steps 50
run chapter19_goal_rl.py --episodes 300 --skip-baseline
run chapter20_marl_basic.py --episodes 200 --grid 9
run chapter21_marl_advanced.py --episodes 150 --batch-size 256

echo
echo "全部运行完成，图像位于 outputs/figures/"
