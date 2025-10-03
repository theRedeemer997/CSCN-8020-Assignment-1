# gridworld.py
# 5x5 deterministic GridWorld used by Value Iteration and Monte Carlo.
# Rewards: regular = -1, grey = -5 at (0,4),(2,2),(3,0), goal = +10 at (4,4).
# Reward is for the *landing* tile. Terminal is absorbing.

import numpy as np

class GridWorld:
    def __init__(self, env_size=5):
        self.env_size = env_size

        # Terminal (goal) at bottom-right (0-indexed row, col)
        self.terminal_state = (4, 4)

        # Actions (dr, dc) — fixed order: Right, Left, Down, Up
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.action_description = ["Right", "Left", "Down", "Up"]

        # Reward map
        self.reward = np.ones((self.env_size, self.env_size), dtype=float) * -1.0
        self.grey_states = [(0, 4), (2, 2), (3, 0)]
        for (r, c) in self.grey_states:
            self.reward[r, c] = -5.0
        self.reward[self.terminal_state] = +10.0

    # Helpers
    def get_size(self):
        return self.env_size

    def in_bounds(self, i, j):
        return 0 <= i < self.env_size and 0 <= j < self.env_size

    def is_terminal_state(self, i, j):
        return (i, j) == self.terminal_state

    # Environment step
    def step(self, action_index, i, j):
        """
        Deterministic transition from (i,j) using action_index.
        If off-grid, stay in place. Reward is for the landing tile.
        Returns: next_i, next_j, reward, done
        """
        # Absorbing terminal
        if self.is_terminal_state(i, j):
            return i, j, self.reward[i, j], True

        dr, dc = self.actions[action_index]
        ni, nj = i + dr, j + dc
        if not self.in_bounds(ni, nj):
            ni, nj = i, j  # bump into wall ⇒ stay

        done = self.is_terminal_state(ni, nj)
        rew = self.reward[ni, nj]
        return ni, nj, rew, done
