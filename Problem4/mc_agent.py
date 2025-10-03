# mc_agent.py
# Monte Carlo Prediction (first-visit) and Monte Carlo Control (epsilon-greedy)
# Works with the GridWorld env you used in Problem 3:
# - rewards on landing
# - terminal at (4,4)
# - actions: Right, Left, Down, Up (indices 0..3)

import numpy as np
import random

class MCAgent:
    def __init__(self, env, gamma=0.9, epsilon=0.1, max_steps=200):
        self.env = env
        self.N = env.get_size()
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.max_steps = int(max_steps)

        # State- and action-value tables
        self.V = np.zeros((self.N, self.N), dtype=float)          # V(s)
        self.Q = np.zeros((self.N, self.N, 4), dtype=float)       # Q(s,a) for 4 actions

        # Running sums/counts for first-visit averaging
        self.returns_sum_V = np.zeros((self.N, self.N), dtype=float)
        self.returns_count_V = np.zeros((self.N, self.N), dtype=int)
        self.returns_sum_Q = np.zeros((self.N, self.N, 4), dtype=float)
        self.returns_count_Q = np.zeros((self.N, self.N, 4), dtype=int)

        # Greedy policy storage (action indices)
        self.policy = np.zeros((self.N, self.N), dtype=int)

        # Operation counters (for your report)
        self.steps_pred = 0   # total env steps during MC Prediction
        self.steps_ctrl = 0   # total env steps during MC Control

    # ---------- Episode generation ----------
    def _random_start_state(self):
        """Uniformly sample a non-terminal start state."""
        while True:
            i = random.randrange(self.N)
            j = random.randrange(self.N)
            if not self.env.is_terminal_state(i, j):
                return i, j

    def _epsilon_greedy_action(self, i, j):
        """ε-greedy over Q(s,·)."""
        if random.random() < self.epsilon:
            return random.randrange(4)  # explore
        return int(np.argmax(self.Q[i, j, :]))  # exploit

    def _policy_action(self, i, j, policy=None):
        """Deterministic action from a given policy array (or self.policy)."""
        pi = self.policy if policy is None else policy
        return int(pi[i, j])

    def generate_episode(self, use_eps_greedy=False, policy=None):
        """
        Generate an episode = list of (state, action, reward).
        - Start from random non-terminal state.
        - Follow ε-greedy wrt Q (if use_eps_greedy=True) else follow given policy.
        - Terminate at goal or after max_steps.
        Reward is for the landing tile (consistent with env.step).
        """
        episode = []
        i, j = self._random_start_state()

        for _ in range(self.max_steps):
            if self.env.is_terminal_state(i, j):
                break

            a = self._epsilon_greedy_action(i, j) if use_eps_greedy else self._policy_action(i, j, policy)
            ni, nj, rew, done = self.env.step(a, i, j)
            episode.append(((i, j), a, rew))
            i, j = ni, nj
            if done:
                break

        return episode

    # ---------- First-Visit MC Prediction (correct first-visit handling) ----------
    def mc_prediction_first_visit(self, episodes=5000, policy=None):
        """
        Estimate V^π using TRUE FIRST-VISIT MC:
        - Build an episode.
        - Compute G_t for every time step by one backward pass.
        - Scan forward; update each state only at its earliest (first) occurrence.
        """
        self.steps_pred = 0  # reset counter

        # Default baseline policy: ALWAYS UP (Problem 2 style). Actions: Right=0, Left=1, Down=2, Up=3
        if policy is None:
            policy = np.full((self.N, self.N), 3, dtype=int)  # 3 = Up

        for _ in range(episodes):
            episode = self.generate_episode(use_eps_greedy=False, policy=policy)
            self.steps_pred += len(episode)

            states = [s for (s, a, r) in episode]
            rewards = [r for (s, a, r) in episode]
            T = len(rewards)
            if T == 0:
                continue

            # Backward pass: compute G_t for every t
            G = 0.0
            G_t = [0.0] * T
            for t in reversed(range(T)):
                G = self.gamma * G + rewards[t]
                G_t[t] = G

            # Forward pass: update only the first occurrence of each state
            seen = set()
            for t, (si, sj) in enumerate(states):
                if (si, sj) in seen:
                    continue
                seen.add((si, sj))
                self.returns_sum_V[si, sj] += G_t[t]
                self.returns_count_V[si, sj] += 1
                self.V[si, sj] = self.returns_sum_V[si, sj] / self.returns_count_V[si, sj]

        return self.V

    # ---------- MC Control (ε-greedy) ----------
    def mc_control_epsilon_greedy(self, episodes=20000):
        """
        Learn Q* and a near-optimal policy with ε-greedy control using first-visit updates.
        """
        self.steps_ctrl = 0  # reset counter

        for _ in range(episodes):
            episode = self.generate_episode(use_eps_greedy=True)
            self.steps_ctrl += len(episode)

            G = 0.0
            visited = set()
            for t in reversed(range(len(episode))):
                (si, sj), a, r = episode[t]
                G = self.gamma * G + r
                if (si, sj, a) not in visited:
                    visited.add((si, sj, a))
                    self.returns_sum_Q[si, sj, a] += G
                    self.returns_count_Q[si, sj, a] += 1
                    self.Q[si, sj, a] = self.returns_sum_Q[si, sj, a] / max(1, self.returns_count_Q[si, sj, a])

        # Greedy policy from learned Q
        for i in range(self.N):
            for j in range(self.N):
                if self.env.is_terminal_state(i, j):
                    self.policy[i, j] = -1
                else:
                    self.policy[i, j] = int(np.argmax(self.Q[i, j, :]))

        # V(s) = max_a Q(s,a) (terminal uses its reward)
        for i in range(self.N):
            for j in range(self.N):
                if self.env.is_terminal_state(i, j):
                    self.V[i, j] = self.env.reward[i, j]
                else:
                    self.V[i, j] = float(np.max(self.Q[i, j, :]))

        return self.V, self.policy

    # ---------- Pretty-print ----------
    def arrows_from_policy(self, policy=None):
        arr = ["→", "←", "↓", "↑"]
        P = self.policy if policy is None else policy
        out = []
        for i in range(self.N):
            row = []
            for j in range(self.N):
                if self.env.is_terminal_state(i, j) or (P[i, j] == -1):
                    row.append(" G ")
                else:
                    row.append(f" {arr[P[i, j]]} ")
            out.append("".join(row))
        return "\n".join(out)
