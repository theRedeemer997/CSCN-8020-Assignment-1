# mc_solved.py
# Problem 4 runner: MC Prediction (Always-Up) + MC Control (epsilon-greedy)
# Prints DP reference (with proper V sync), MC tables/policies, and operation counters.

import time
import random
import numpy as np

from gridworld import GridWorld
from value_iteration_agent import ValueIterationAgent
from mc_agent import MCAgent

# ---------- Reproducibility ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------- Settings ----------
ENV_SIZE = 5
GAMMA = 0.9
EPS = 0.1
MAX_STEPS = 200
PRED_EPISODES = 5000
CTRL_EPISODES = 30000
ARROWS = ["→", "←", "↓", "↑"]  # Right=0, Left=1, Down=2, Up=3

# ---------- Helpers ----------
def print_value_table(V, title):
    print(f"\n{title}")
    for r in range(V.shape[0]):
        print(" ".join(f"{V[r, c]:6.2f}" for c in range(V.shape[1])))

def print_policy_arrows_from_indices(env, policy_idx, title):
    print(f"\n{title}")
    for i in range(env.get_size()):
        row = []
        for j in range(env.get_size()):
            if env.is_terminal_state(i, j):
                row.append(" G ")
            else:
                row.append(f" {ARROWS[policy_idx[i, j]]} ")
        print("".join(row))

# ---------- Build env + DP reference (with sync to agent V) ----------
env = GridWorld(ENV_SIZE)
dp = ValueIterationAgent(env, gamma=GAMMA, theta_threshold=1e-9)

# Run batch value iteration to compute a DP reference V*
V = np.zeros((ENV_SIZE, ENV_SIZE), dtype=float)
for _ in range(10_000):
    # IMPORTANT: sync the agent's internal V so calculate_max_value uses CURRENT V
    dp.update_value_function(V)

    Vn = V.copy()
    for i in range(ENV_SIZE):
        for j in range(ENV_SIZE):
            if env.is_terminal_state(i, j):
                Vn[i, j] = env.reward[i, j]   # keep terminal at +10
            else:
                best_v, _, _ = dp.calculate_max_value(i, j)  # max_a [ R(next)+gamma*V(next) ]
                Vn[i, j] = best_v

    if np.max(np.abs(Vn - V)) <= 1e-9:
        V = Vn
        break
    V = Vn

dp.update_value_function(V)
dp.update_greedy_policy()

print_value_table(V, "DP Optimal Value Function V* (reference)")
print("\nDP Greedy Policy (arrows):")
dp.print_policy()

# ---------- MC Prediction (first-visit) : Always-Up baseline ----------
mc = MCAgent(env, gamma=GAMMA, epsilon=EPS, max_steps=MAX_STEPS)

t0 = time.perf_counter()
# policy_idx=None => default baseline "Always Up" (action index 3) inside MCAgent
V_pi = mc.mc_prediction_first_visit(episodes=PRED_EPISODES, policy_idx=None)
t_pred_ms = (time.perf_counter() - t0) * 1000.0

print_value_table(V_pi, f"\nMC Prediction V^π (Always Up), {PRED_EPISODES} episodes — {t_pred_ms:.1f} ms")
print(f"[MC Prediction] episodes={PRED_EPISODES}, total steps={mc.steps_pred}, "
      f"avg length={mc.steps_pred/max(1,PRED_EPISODES):.2f} steps/episode")

# ---------- MC Control (epsilon-greedy) ----------
t0 = time.perf_counter()
V_star_mc, pi_mc = mc.mc_control_epsilon_greedy(episodes=CTRL_EPISODES)
t_ctrl_ms = (time.perf_counter() - t0) * 1000.0

print_value_table(V_star_mc, f"\nMC Control V* (approx), {CTRL_EPISODES} episodes — {t_ctrl_ms:.1f} ms")
print(f"[MC Control] episodes={CTRL_EPISODES}, total steps={mc.steps_ctrl}, "
      f"avg length={mc.steps_ctrl/max(1,CTRL_EPISODES):.2f} steps/episode")

print_policy_arrows_from_indices(env, pi_mc, "MC Control Greedy Policy (arrows)")

print("\nDP Greedy Policy (for visual comparison):")
dp.print_policy()

# ---------- Optional: quick numeric closeness summary ----------
max_abs_diff = float(np.max(np.abs(V_star_mc - V)))
mean_abs_diff = float(np.mean(np.abs(V_star_mc - V)))
print(f"\n[Summary] MC-Control vs DP — max |V_mc - V_dp| = {max_abs_diff:.3f}, mean |V_mc - V_dp| = {mean_abs_diff:.3f}")
