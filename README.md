# CSCN8020 — Assignment 1 (MDP • DP • MC)

A compact, reproducible walkthrough of Problems 1–4 on a 5×5 Gridworld, using:

- **Problem 1:** MDP modeling (states, actions, δ transitions, rewards, γ)
- **Problem 2:** Value Iteration (step-by-step updates for two iterations)
- **Problem 3:** Dynamic Programming (Batch & In-Place VI) → \(V^_, \pi^_\)
- **Problem 4:** Monte Carlo (First-Visit Prediction & ε-Greedy Control) + DP comparison

---

## Solutions

- The solution of `Problem1` can be found in file inside with the name `ANSWER.md`
- The solution of `Problem2` can be found in file inside with the name `ANSWERTWO.md`
- The solution of `Problem3` can be found in file inside with the name`solution.md`
- The solution of `Problem4` can be found in file inside with the name `solution.md`

## Setup & Run

```bash
# (optional) create venv
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Problem 3 (Value Iteration):
python value_iteration_solved.py

# Problem 4 (Monte Carlo):
python mc_solved.py

```
