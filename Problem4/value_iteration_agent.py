# value_iteration_agent.py
# Holds the value table V and provides one-step lookahead + greedy policy utilities.

import numpy as np

class ValueIterationAgent:
    def __init__(self, env, gamma=0.9, theta_threshold=1e-9):
        self.env = env
        self.env_size = env.get_size()
        self.gamma = float(gamma)
        self.theta_threshold = float(theta_threshold)

        # Value function V(s)
        self.V = np.zeros((self.env_size, self.env_size), dtype=float)

        # Greedy policy (action indices) and display arrows
        self.pi_idx = np.zeros((self.env_size, self.env_size), dtype=int)
        self.arrows = ["→", "←", "↓", "↑"]  # matches env.actions: Right, Left, Down, Up

    # Accessors
    def get_value_function(self):
        return self.V

    def update_value_function(self, new_V):
        self.V = np.array(new_V, dtype=float, copy=True)

    def is_done(self, new_V):
        """Converged when max|V - new_V| <= theta."""
        return np.max(np.abs(self.V - new_V)) <= self.theta_threshold

    # One-step lookahead: max_a [ R(next) + gamma * V(next) ]
    def calculate_max_value(self, i, j):
        best_val, best_idx = -1e18, 0
        for a_idx, _ in enumerate(self.env.actions):
            ni, nj, rew, done = self.env.step(a_idx, i, j)
            q = rew + (0.0 if done else self.gamma * self.V[ni, nj])
            if q > best_val:
                best_val, best_idx = q, a_idx
        return best_val, best_idx, self.env.action_description[best_idx]

    # Greedy policy from V
    def update_greedy_policy(self):
        for i in range(self.env_size):
            for j in range(self.env_size):
                if self.env.is_terminal_state(i, j):
                    self.pi_idx[i, j] = -1
                else:
                    _, a_idx, _ = self.calculate_max_value(i, j)
                    self.pi_idx[i, j] = a_idx

    def print_policy(self):
        """Pretty-print arrows; goal shown as ' G '."""
        print("\nGreedy Policy (arrows):")
        for i in range(self.env_size):
            row = []
            for j in range(self.env_size):
                if self.env.is_terminal_state(i, j):
                    row.append(" G ")
                else:
                    row.append(f" {self.arrows[self.pi_idx[i, j]]} ")
            print("".join(row))
