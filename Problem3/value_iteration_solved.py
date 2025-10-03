# value_iteration_solved.py
# Value Iteration (batch + in-place) on 5x5 GridWorld
# Prints: numeric tables, terminal heatmaps (ANSI), and greedy policy arrows.
# Terminal state's value is pinned to 0.0 (reward still gained on landing in terminal).

import time
import numpy as np

from gridworld import GridWorld
from value_iteration_agent import ValueIterationAgent

ENV_SIZE = 5
GAMMA = 0.9
THETA_THRESHOLD = 1e-9
MAX_ITERATIONS = 10_000

def print_value_table(V, title):
    print(f"\n{title}")
    for r in range(V.shape[0]):
        print(" ".join(f"{V[r, c]:6.2f}" for c in range(V.shape[1])))

def print_terminal_heatmap(V, title):
    """
    ANSI 256-color heatmap (no GUI). Uses grayscale background 232..255.
    Also prints the numbers to the right of the color blocks.
    """
    vmin, vmax = float(np.min(V)), float(np.max(V))
    span = vmax - vmin if vmax != vmin else 1.0

    def code(x):
        # map value to 232..255 (24 levels)
        t = (x - vmin) / span
        return 232 + int(round(t * 23))

    print(f"\n{title} (terminal heatmap)")
    # header for columns
    print("    " + " ".join(f" c{j} " for j in range(V.shape[1])))
    for i in range(V.shape[0]):
        blocks = []
        for j in range(V.shape[1]):
            c = code(V[i, j])
            blocks.append(f"\x1b[48;5;{c}m  \x1b[0m")  # two spaces with bg color
        # append numeric values for readability
        nums = " ".join(f"{V[i, j]:5.2f}" for j in range(V.shape[1]))
        print(f"r{i}  " + "".join(blocks) + "   " + nums)

def run_value_iteration_batch(env, agent):
    iters = 0
    while iters < MAX_ITERATIONS:
        V_old = agent.get_value_function()
        new_V = np.copy(V_old)
        for i in range(env.get_size()):
            for j in range(env.get_size()):
                if env.is_terminal_state(i, j):
                    new_V[i, j] = 0.0
                else:
                    best_v, _, _ = agent.calculate_max_value(i, j)
                    new_V[i, j] = best_v
        if agent.is_done(new_V):
            agent.update_value_function(new_V)
            iters += 1
            break
        agent.update_value_function(new_V)
        iters += 1
    return iters

def run_value_iteration_inplace(env, agent):
    iters = 0
    while iters < MAX_ITERATIONS:
        delta = 0.0
        for i in range(env.get_size()):
            for j in range(env.get_size()):
                old = agent.V[i, j]
                if env.is_terminal_state(i, j):
                    agent.V[i, j] = 0.0
                else:
                    best_v, _, _ = agent.calculate_max_value(i, j)
                    agent.V[i, j] = best_v
                delta = max(delta, abs(agent.V[i, j] - old))
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
    V_batch = agent.get_value_function()
    print_value_table(V_batch, f"Optimal Value Function (Batch) — {it_batch} iterations, {t_batch:.1f} ms")
    print_terminal_heatmap(V_batch, "VI (Batch) — V*")
    agent.update_greedy_policy()
    agent.print_policy()

    # -------- In-place VI --------
    print("\n=== In-Place (Gauss–Seidel) Value Iteration ===")
    agent.update_value_function(np.zeros_like(agent.get_value_function()))
    t0 = time.perf_counter()
    it_inplace = run_value_iteration_inplace(env, agent)
    t_inplace = (time.perf_counter() - t0) * 1000.0
    V_inplace = agent.get_value_function()
    print_value_table(V_inplace, f"Optimal Value Function (In-place) — {it_inplace} iterations, {t_inplace:.1f} ms")
    print_terminal_heatmap(V_inplace, "VI (In-place) — V*")
    agent.update_greedy_policy()
    agent.print_policy()

    # Sanity check
    agent.update_value_function(np.zeros_like(agent.get_value_function()))
    run_value_iteration_batch(env, agent); Vb = agent.get_value_function().copy()
    agent.update_value_function(np.zeros_like(agent.get_value_function()))
    run_value_iteration_inplace(env, agent); Vi = agent.get_value_function().copy()
    print(f"\nMax |V_batch - V_inplace| = {np.max(np.abs(Vb - Vi)):.3e}")

if __name__ == "__main__":
    main()
