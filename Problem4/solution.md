# Problem 4 — Monte Carlo Methods on 5×5 Gridworld

## Environment (same as Problem 3)

- Grid: 5×5
- Actions (indices): Right=0, Left=1, Down=2, Up=3 (deterministic; bump ⇒ stay)
- Rewards (on landing):
  - Regular: −1
  - Grey: −5 at (0,4), (2,2), (3,0)
  - Goal (terminal): +10 at (4,4)
- Discount: gamma = 0.9

**Files submitted**

- `gridworld.py` — environment and reward map
- `value_iteration_agent.py` — DP helper used for the reference (V\* and greedy policy)
- `mc_agent.py` — Monte Carlo agent: first-visit MC prediction + ε-greedy MC control
- `mc_solved.py` — runner that prints all tables/policies + operation counters

**How to reproduce**

Go to `Problem3` folder

```bash
python mc_solved.py
```

# Monte Carlo vs. Value Iteration — Comparison

## Quick comparison (your runs)

| Method                        |                    Time |                                                 Work / Operations |   Episodes | Result quality                                                             |
| ----------------------------- | ----------------------: | ----------------------------------------------------------------: | ---------: | -------------------------------------------------------------------------- | --- | --------------------------------------- | ----------------- | -------------------------------------- |
| **Value Iteration (DP)**      | **≈ 0.4 ms** (9 sweeps) |                                                     Per sweep: \( |          S |                                                                            | A   | = 25×4 = 100\) Q-evals → **~900** total | N/A (model-based) | Ground-truth \(V^\*\) / optimal policy |
| **MC Prediction** (Always-Up) |            **450.2 ms** | **1,000,000** env steps (avg **200.00** steps/ep × **5,000** eps) |  **5,000** | \(V^\pi\) for a bad policy (strong negatives near top/grey)                |
| **MC Control** (ε=0.1)        |            **227.5 ms** |             **142,685** env steps (avg **4.76** × **30,000** eps) | **30,000** | Approx \(V^\*\); **max diff** vs DP = **1.175**, **mean diff** = **0.500** |

**Takeaway:** On this small 5×5 grid with a known model, **DP is fastest and exact**. MC Control is close but needs **many** samples.

---

## Computational complexity (big-O)

- **Value Iteration:** per sweep \( \Theta(|S||A|) \). Converges in ~\(O\!\big(\tfrac{1}{1-\gamma}\log \tfrac{1}{\varepsilon}\big)\) sweeps to tolerance \(\varepsilon\).
- **Monte Carlo (Prediction/Control):** \( \Theta(\text{episodes} \times \text{avg episode length}) \). Variance decreases with more episodes; no model needed.

---

## What stands out in your outputs

- **Speed:** DP did ~**900** Q-evaluations in ~**0.4 ms**.  
  MC Control needed **142,685** environment steps in **~227.5 ms**.
- **Accuracy:** DP is optimal; MC Control is **close** but not identical at 30k episodes (max \(|V*{\text{MC}}-V*{\text{DP}}|=1.175\), mean \(=0.500\)).
- **Episodes:** DP uses **no episodes** (model-based).  
  MC Prediction: **5k** long episodes (bad baseline → often hits max length).  
  MC Control: **30k** short episodes (good learned policy → quick termination).
- **Memory:** DP stores \(V(s)\) (\(|S|\)). MC Control stores \(Q(s,a)\) (\(|S||A|\)) plus counters (still tiny here).
- **Assumptions:** DP **requires the model** (rewards & transitions). MC learns **from experience** (no model).
- **Convergence behavior:** DP is a contraction mapping → deterministic convergence.  
  MC is unbiased but **high-variance**; with fixed ε it’s near-optimal; with **GLIE** (decaying ε) it converges.

---

## How to make MC match DP more closely

- Increase control episodes (e.g., **100k+**).
- Use **decaying ε** (e.g., \(0.1 \to 0.01\)).
- Consider **every-visit** MC (often lower variance than first-visit).
- Ensure reasonable episode cap (`max_steps`) to avoid overly long rollouts.

---

### Bottom line

- With a **known model** and small \(|S|\), **Value Iteration** is the fastest path to \(V^_\) and \(\pi^_\).
- When the **model is unknown**, **Monte Carlo** is slower/noisier but still learns toward the DP optimum given sufficient data.
