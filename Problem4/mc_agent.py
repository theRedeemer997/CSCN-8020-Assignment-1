# mc_agent.py
# Monte Carlo Prediction (first-visit) and Monte Carlo Control (epsilon-greedy)
# Works with GridWorld from Problem 3 (rewards on landing; terminal at (4,4)).

import numpy as np
import random

class MCAgent:
    def __init__(self, env, gamma=0.9, epsilon=0.1, max_steps=200):
        self.env = env
        self.N = env.get_size()
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.max_steps = int(max_steps)

        # State-value (prediction) and action-value (control) tables
        self.V = np.zeros((self.N, self.N), dtype=float)
        self.Q = np.zeros((self.N, self.N, 4), dtype=float)  # actions: Right=0, Left=1, Down=2, Up=3
        self.policy = np.zeros((self.N, self.N), dtype=int)  # greedy indices after control

        # Operation counters
        self.steps_pred = 0
        self.steps_ctrl = 0

    # ---------------- Helpers ----------------
    def _random_start_state(self):
        while True:
            i = random.randrange(self.N)
            j = random.randrange(self.N)
            if not self.env.is_terminal_state(i, j):
                return i, j

    def _epsilon_greedy_action(self, i, j):
        if random.random() < self.epsilon:
            return random.randrange(4)
        return int(np.argmax(self.Q[i, j, :]))

    def _policy_action(self, i, j, policy_idx):
        return int(policy_idx[i, j])

    def generate_episode(self, use_eps_greedy=False, policy_idx=None):
        """
        Return a list of (state, action, reward) with rewards on landing.
        Stops at terminal or max_steps.
        """
        s_i, s_j = self._random_start_state()
        episode = []
        for _ in range(self.max_steps):
            if self.env.is_terminal_state(s_i, s_j):
                break
            a = self._epsilon_greedy_action(s_i, s_j) if use_eps_greedy else self._policy_action(s_i, s_j, policy_idx)
            ni, nj, rew, done = self.env.step(a, s_i, s_j)
            episode.append(((s_i, s_j), a, rew))
            s_i, s_j = ni, nj
            if done:
                break
        return episode

    # ------------- MC Prediction (first-visit) -------------
    def mc_prediction_first_visit(self, episodes=5000, policy_idx=None):
        """
        Estimate V^π for a deterministic policy π (given as indices).
        If policy_idx is None, default to Problem-2 baseline: ALWAYS UP (action index 3).
        """
        self.steps_pred = 0
        if policy_idx is None:
            policy_idx = np.full((self.N, self.N), 3, dtype=int)  # Up = 3

        returns_sum = np.zeros((self.N, self.N), dtype=float)
        returns_count = np.zeros((self.N, self.N), dtype=int)
        self.V.fill(0.0)

        for _ in range(episodes):
            ep = self.generate_episode(use_eps_greedy=False, policy_idx=policy_idx)
            self.steps_pred += len(ep)
            G = 0.0
            visited = set()
            for t in reversed(range(len(ep))):
                (si, sj), a, r = ep[t]
                G = self.gamma * G + r
                if (si, sj) not in visited:
                    visited.add((si, sj))
                    returns_sum[si, sj] += G
                    returns_count[si, sj] += 1
                    self.V[si, sj] = returns_sum[si, sj] / max(1, returns_count[si, sj])

        # Keep terminal consistent with env (reward on landing)
        ti, tj = self.env.terminal_state
        self.V[ti, tj] = self.env.reward[ti, tj]
        return self.V

    # ------------- MC Control (epsilon-greedy) -------------
    def mc_control_epsilon_greedy(self, episodes=30000):
        """
        Learn Q* with epsilon-greedy exploring starts, then return greedy policy and V from Q.
        """
        self.steps_ctrl = 0
        self.Q.fill(0.0)
        self.policy.fill(0)
        returns_sum_Q = np.zeros_like(self.Q)
        returns_count_Q = np.zeros_like(self.Q)

        for _ in range(episodes):
            ep = self.generate_episode(use_eps_greedy=True)
            self.steps_ctrl += len(ep)
            G = 0.0
            visited = set()
            for t in reversed(range(len(ep))):
                (si, sj), a, r = ep[t]
                G = self.gamma * G + r
                if (si, sj, a) not in visited:
                    visited.add((si, sj, a))
                    returns_sum_Q[si, sj, a] += G
                    returns_count_Q[si, sj, a] += 1
                    self.Q[si, sj, a] = returns_sum_Q[si, sj, a] / max(1, returns_count_Q[si, sj, a])

        # Greedy policy + V(s)=max_a Q(s,a)
        for i in range(self.N):
            for j in range(self.N):
                if self.env.is_terminal_state(i, j):
                    self.policy[i, j] = -1
                    self.V[i, j] = self.env.reward[i, j]
                else:
                    self.policy[i, j] = int(np.argmax(self.Q[i, j, :]))
                    self.V[i, j] = float(np.max(self.Q[i, j, :]))

        return self.V, self.policy
