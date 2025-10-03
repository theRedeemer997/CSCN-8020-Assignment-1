# value_iteration_agent.py
# Agent that holds the value table V and provides one-step lookahead and greedy policy utilities.

import numpy as np

class ValueIterationAgent:
    def __init__(self, env, gamma=0.9, theta_threshold=1e-9):
        self.env = env
        self.env_size = env.get_size()
        self.gamma = float(gamma)
        self.theta_threshold = float(theta_threshold)

        # Value function table V(s) initialized to zeros
        self.V = np.zeros((self.env_size, self.env_size), dtype=float)

        # Greedy policy storage (indices/arrows for printing)
        self.pi_idx = np.zeros((self.env_size, self.env_size), dtype=int)
        self.arrows = ["→", "←", "↓", "↑"]  # matches env.actions order

    # ----- basic accessors -----
    def get_value_function(self):
        return self.V

    def update_value_function(self, new_V):
        self.V = np.array(new_V, dtype=float, copy=True)

    def is_done(self, new_V):
        """Stop when the largest absolute change is below theta."""
        diff = np.max(np.abs(self.V - new_V))
        return diff <= self.theta_threshold

    # ----- core: one-step lookahead -----
    def calculate_max_value(self, i, j):
        """
        Compute max_a Q(s,a) for state (i,j), and also return the argmax and its name.
        Q(s,a) = reward(next) + gamma * V(next), but if next is terminal → no future term.
        Returns: (best_value, best_action_index, best_action_name)
        """
        best_val, best_idx = -1e18, 0
        for a_idx, _ in enumerate(self.env.actions):
            ni, nj, rew, done = self.env.step(a_idx, i, j)
            q = rew + (0.0 if done else self.gamma * self.V[ni, nj])
            if q > best_val:
                best_val, best_idx = q, a_idx
        best_name = self.env.action_description[best_idx]
        return best_val, best_idx, best_name

    # ----- greedy policy from current V -----
    def update_greedy_policy(self):
        for i in range(self.env_size):
            for j in range(self.env_size):
                if self.env.is_terminal_state(i, j):
                    self.pi_idx[i, j] = -1  # goal marker
                else:
                    _, a_idx, _ = self.calculate_max_value(i, j)
                    self.pi_idx[i, j] = a_idx

    def print_policy(self):
        """Pretty-print greedy arrows; goal shown as ' G '."""
        print("\nGreedy Policy (arrows):")
        for i in range(self.env_size):
            row = []
            for j in range(self.env_size):
                if self.env.is_terminal_state(i, j):
                    row.append(" G ")
                else:
                    a = self.pi_idx[i, j]
                    row.append(f" {self.arrows[a]} ")
            print("".join(row))
