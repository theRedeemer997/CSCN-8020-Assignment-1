# mc_solved.py
# Runs Monte Carlo Prediction (first-visit) and Monte Carlo Control (epsilon-greedy)
# on the same 5x5 GridWorld. Prints:
# - MC Prediction V^π for a baseline policy (default: Always Up),
# - MC Control V* (approx) and greedy policy,
# - DP greedy policy (from VI) for visual comparison,
# - Operation counters: total env steps and average episode length,
# - MAE metrics: MC Prediction vs DP policy-evaluation (same policy),
#                MC Control V* vs DP V* (VI).

import time
import numpy as np
from gridworld import GridWorld
from value_iteration_agent import ValueIterationAgent   # DP for comparison
from mc_agent import MCAgent

ENV_SIZE = 5
GAMMA = 0.9
EPS = 0.1
PRED_EPISODES = 5000
CTRL_EPISODES = 30000  # raise to 100000+ for closer match to DP

def print_value_table(V, title):
    print(f"\n{title}")
    for r in range(V.shape[0]):
        print(" ".join(f"{V[r, c]:6.2f}" for c in range(V.shape[1])))

def policy_evaluation(env, policy, gamma=0.9, theta=1e-9):
    """
    Iterative policy evaluation for a deterministic policy (array of action indices).
    Convention for comparison with MC Prediction: terminal state's value = 0.
    """
    n = env.get_size()
    V = np.zeros((n, n), dtype=float)
    while True:
        delta = 0.0
        newV = V.copy()
        for i in range(n):
            for j in range(n):
                if env.is_terminal_state(i, j):
                    newV[i, j] = 0.0  # terminal convention for prediction MAE
                    continue
                a = int(policy[i, j])
                ni, nj, r, done = env.step(a, i, j)
                newV[i, j] = r + (0.0 if done else gamma * V[ni, nj])
                delta = max(delta, abs(newV[i, j] - V[i, j]))
        V = newV
        if delta <= theta:
            break
    return V

def main():
    env = GridWorld(ENV_SIZE)

    # === DP optimal (for control MAE & arrows) ===
    dp = ValueIterationAgent(env, gamma=GAMMA, theta_threshold=1e-9)
    V = np.zeros((ENV_SIZE, ENV_SIZE), dtype=float)
    for _ in range(10_000):
        newV = V.copy()
        for i in range(ENV_SIZE):
            for j in range(ENV_SIZE):
                if env.is_terminal_state(i, j):
                    newV[i, j] = env.reward[i, j]
                else:
                    best_v, _, _ = dp.calculate_max_value(i, j)
                    newV[i, j] = best_v
        if np.max(np.abs(newV - V)) < 1e-9:
            V = newV
            break
        V = newV
    dp.update_value_function(V)
    dp.update_greedy_policy()
    V_dp_opt = dp.get_value_function()  # DP V* (for control MAE)

    # === MC Prediction (first-visit) ===
    print("\n=== Monte Carlo Prediction (first-visit) for a fixed baseline policy ===")
    mc = MCAgent(env, gamma=GAMMA, epsilon=EPS, max_steps=200)

    t0 = time.perf_counter()
    # Baseline default in agent is ALWAYS UP; pass policy=None to use it.
    V_pi = mc.mc_prediction_first_visit(episodes=PRED_EPISODES, policy=None)
    t_pred = (time.perf_counter() - t0) * 1000.0

    print_value_table(V_pi, f"MC Prediction V^π (first-visit), {PRED_EPISODES} episodes — {t_pred:.1f} ms")
    steps_pred = mc.steps_pred
    avg_len_pred = steps_pred / max(1, PRED_EPISODES)
    print(f"\n[MC Prediction] episodes={PRED_EPISODES}, total env steps={steps_pred}, avg length={avg_len_pred:.2f} steps/episode")

    # MAE: MC Prediction vs DP policy evaluation (same policy)
    policy_up = np.full((ENV_SIZE, ENV_SIZE), 3, dtype=int)  # always up
    V_pi_dp = policy_evaluation(env, policy_up, gamma=GAMMA, theta=1e-9)
    mask_nonterm = np.ones((ENV_SIZE, ENV_SIZE), dtype=bool)
    ti, tj = env.terminal_state
    mask_nonterm[ti, tj] = False
    mae_pred = np.mean(np.abs(V_pi[mask_nonterm] - V_pi_dp[mask_nonterm]))
    print(f"[MAE] MC Prediction vs DP (same policy, non-terminal states): {mae_pred:.4f}")

    # === MC Control (ε-greedy) ===
    print("\n=== Monte Carlo Control (epsilon-greedy) ===")
    t0 = time.perf_counter()
    V_star, pi_star = mc.mc_control_epsilon_greedy(episodes=CTRL_EPISODES)
    t_ctrl = (time.perf_counter() - t0) * 1000.0

    print_value_table(V_star, f"MC Control V* (approx), {CTRL_EPISODES} episodes — {t_ctrl:.1f} ms")
    print("\nMC Control Greedy Policy (arrows):")
    print(mc.arrows_from_policy(pi_star))

    steps_ctrl = mc.steps_ctrl
    avg_len_ctrl = steps_ctrl / max(1, CTRL_EPISODES)
    print(f"\n[MC Control] episodes={CTRL_EPISODES}, total env steps={steps_ctrl}, avg length={avg_len_ctrl:.2f} steps/episode")

    # MAE: MC Control V* vs DP V*
    mae_ctrl = np.mean(np.abs(V_star - V_dp_opt))
    print(f"[MAE] MC Control V* vs DP V*: {mae_ctrl:.4f}")

    print("\nDP Greedy Policy (for visual comparison):")
    dp.print_policy()

    print("\n(Note) Monte Carlo is sample-based; with more episodes (and possibly decaying ε), "
          "its V* and arrows approach the DP solution.)")

if __name__ == "__main__":
    main()
