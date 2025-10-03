# value_iteration_solved.py
# Runs Value Iteration on the GridWorld using both:
#   (1) Batch (synchronous) VI   and   (2) In-place (Gauss–Seidel) VI
# Prints V* and the greedy policy for both; they should match.

import time
import numpy as np
from gridworld import GridWorld
from value_iteration_agent import ValueIterationAgent

# -------------------
# Hyperparameters
# -------------------
ENV_SIZE = 5            # 5x5 grid
GAMMA = 0.9             # discount factor
THETA_THRESHOLD = 1e-9  # convergence tolerance
MAX_ITERATIONS = 10_000 # safety cap

def print_value_table(V, title):
    print(f"\n{title}")
    for r in range(V.shape[0]):
        print(" ".join(f"{V[r, c]:6.2f}" for c in range(V.shape[1])))

def run_value_iteration_batch(env, agent):
    """
    Batch (synchronous) Value Iteration:
    Use a COPY new_V each sweep; do not reuse within-sweep updates.
    new_V[i,j] = max_a { R(next) + gamma * V_old(next) }.
    """
    iters = 0
    while iters < MAX_ITERATIONS:
        V_old = agent.get_value_function()
        new_V = np.copy(V_old)

        for i in range(env.get_size()):
            for j in range(env.get_size()):
                if env.is_terminal_state(i, j):
                    new_V[i, j] = env.reward[i, j]   # keep terminal fixed at +10
                else:
                    best_v, _, _ = agent.calculate_max_value(i, j)
                    new_V[i, j] = best_v

        if agent.is_done(new_V):         # max |new_V - V_old| <= theta?
            agent.update_value_function(new_V)
            iters += 1
            break

        agent.update_value_function(new_V)
        iters += 1

    return iters

def run_value_iteration_inplace(env, agent):
    """
    In-place (Gauss–Seidel) Value Iteration:
    Update agent.V directly; reuse fresh values within the same sweep.
    """
    iters = 0
    while iters < MAX_ITERATIONS:
        delta = 0.0
        for i in range(env.get_size()):
            for j in range(env.get_size()):
                old = agent.V[i, j]
                if env.is_terminal_state(i, j):
                    agent.V[i, j] = env.reward[i, j]
                else:
                    best_v, _, _ = agent.calculate_max_value(i, j)
                    agent.V[i, j] = best_v
                dv = abs(agent.V[i, j] - old)
                if dv > delta:
                    delta = dv
        iters += 1
        if delta <= THETA_THRESHOLD:
            break
    return iters

def main():
    env = GridWorld(ENV_SIZE)
    agent = ValueIterationAgent(env, GAMMA, THETA_THRESHOLD)

    # -------- Batch VI --------
    print("=== Batch (Synchronous) Value Iteration ===")
    t0 = time.perf_counter()
    it_batch = run_value_iteration_batch(env, agent)
    t_batch = (time.perf_counter() - t0) * 1000.0

    print_value_table(agent.get_value_function(),
                      f"Optimal Value Function (Batch) — {it_batch} iterations, {t_batch:.1f} ms")
    agent.update_greedy_policy()
    agent.print_policy()

    # -------- In-place VI --------
    print("\n=== In-Place (Gauss–Seidel) Value Iteration ===")
    agent.update_value_function(np.zeros_like(agent.get_value_function()))
    t0 = time.perf_counter()
    it_inplace = run_value_iteration_inplace(env, agent)
    t_inplace = (time.perf_counter() - t0) * 1000.0

    print_value_table(agent.get_value_function(),
                      f"Optimal Value Function (In-place) — {it_inplace} iterations, {t_inplace:.1f} ms")
    agent.update_greedy_policy()
    agent.print_policy()

    # -------- Sanity check (both should match) --------
    agent.update_value_function(np.zeros_like(agent.get_value_function()))
    run_value_iteration_batch(env, agent); Vb = agent.get_value_function().copy()
    agent.update_value_function(np.zeros_like(agent.get_value_function()))
    run_value_iteration_inplace(env, agent); Vi = agent.get_value_function().copy()
    print(f"\nMax |V_batch - V_inplace| = {np.max(np.abs(Vb - Vi)):.3e}")

if __name__ == "__main__":
    main()
