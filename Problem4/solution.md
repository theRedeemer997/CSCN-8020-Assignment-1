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

## DP Reference

## V* (from value iteration)
|       |       |       |       |       |
|-------|-------|-------|-------|-------|
| -0.43 |  0.63 |  1.81 |  3.12 |  4.58 |
|  0.63 |  1.81 |  3.12 |  4.58 |  6.20 |
|  1.81 |  3.12 |  4.58 |  6.20 |  8.00 |
|  3.12 |  4.58 |  6.20 |  8.00 | 10.00 |
|  4.58 |  6.20 |  8.00 | 10.00 | 10.00 |

## Greedy policy (arrows)
| → | → | → | ↓ | ↓ |
|---|---|---|---|---|
| → | → | → | → | ↓ |
| → | ↓ | → | → | ↓ |
| → | → | → | → | ↓ |
| → | → | → | → | G |

## Task 1 — MC Prediction (First-Visit)

Policy: Always Up (action index 3)

Episodes: 5,000 Time: 840.8 ms

Operations: total env steps = 1,000,000 · avg length = 200.00 steps/episode

Estimated 
V
<sup>π</sup>

| -1.00  | -1.00  | -1.00  | -1.00  | -5.00  |
|--------|--------|--------|--------|--------|
| -10.00 | -10.00 | -10.00 | -10.00 | -50.00 |
| -10.00 | -10.00 | -10.00 | -10.00 | -46.00 |
| -10.00 | -10.00 | -14.00 | -10.00 | -42.40 |
| -14.00 | -10.00 | -13.60 | -10.00 |  10.00 |

`Interpretation`: With “Always Up,” top states often bump the wall (repeated −1), and the grey at (0,4) accumulates ≈ −5/(1−0.9)=−50. MC Prediction reflects the value of this fixed (poor) policy.

## Task 2 — MC Control (ε-greedy)
- ε: 0.1

- Episodes: 30,000 Time: 468.4 ms

- Operations: total env steps = 142,685 · avg length = 4.76 steps/episode

## Approximate 
V∗
 (from MC Control):

 | -1.34 | -0.24 |  0.63 |  2.42 |  4.05 |
|-------|-------|-------|-------|-------|
| -0.29 |  0.97 |  2.44 |  4.00 |  5.85 |
|  0.98 |  2.43 |  4.14 |  5.89 |  7.85 |
|  2.54 |  4.08 |  5.91 |  7.86 | 10.00 |
|  4.09 |  5.84 |  7.83 | 10.00 | 10.00 |

## Greedy policy learned (arrows):
| → | ↓ | ↓ | ↓ | ↓ |
|---|---|---|---|---|
| ↓ | ↓ | → | → | ↓ |
| → | ↓ | ↓ | ↓ | ↓ |
| ↓ | → | → | → | ↓ |
| → | → | → | → | G |



```markdown
**Numerical closeness (MC Control vs DP values):** max |V_MC − V_DP| = 1.175 · mean |V_MC − V_DP| = 0.500
```

## Performance & Complexity Comparison


| Method                        | Time                    | Operations                          | Episodes   | Notes                                                  |
|-------------------------------|-------------------------|-------------------------------------|------------|--------------------------------------------------------|
| **Value Iteration (DP)**      | ≈ **0.4 ms** (9 sweeps)* | Per sweep: \|S\|\|A\|=25×4=100 ⇒ **~900** Q-evals total | N/A        | Exact V<sup>\*</sup>, optimal policy (requires model)   |
| **MC Prediction** (Always-Up) | **840.8 ms**            | **1,000,000** env steps             | **5,000**  | Estimates V<sup>π</sup> for a fixed (poor) policy       |
| **MC Control** (ε=0.1)        | **468.4 ms**            | **142,685** env steps               | **30,000** | Near-optimal V<sup>\*</sup> & greedy policy learned from experience |

\* DP timing shown here is from a reference run; wall-clock varies by machine.

### Asymptotic complexity (big-O)

- **Value Iteration (DP):** per sweep Θ(\|S\|\|A\|). Converges in about O(1/(1-γ)log(1/ε)) sweeps to tolerance ε.
- **Monte Carlo (Prediction/Control):** Θ(episodes × avg episode length). Unbiased but higher variance; no transition/reward model needed.

### Observations (from your run)

- **Speed:** DP finishes in a few sweeps; MC needs many sampled steps (trajectories).
- **Accuracy:** DP is exact; MC Control is close and improves with more episodes or decaying ε.
- **Memory:** DP stores V(s) (size \|S\|); MC Control stores Q(s,a) (size \|S\|\|A\|) plus counters.