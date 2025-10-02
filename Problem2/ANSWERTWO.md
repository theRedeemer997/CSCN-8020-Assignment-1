# Problem 2 — 2×2 Gridworld

**Given**

- States: S = {s1, s2, s3, s4}
- Actions: A = {up, down, left, right}
- Transitions: valid moves are deterministic; invalid (wall) ⇒ stay in the same state
- Rewards: R(s1)=5, R(s2)=10, R(s3)=1, R(s4)=2 (same for all actions)
- Discount: γ = 0.9 (assumed)

**Next-state map (deterministic; bump = stay)**

- s1: up→s1, left→s1, right→s2, down→s3
- s2: up→s2, right→s2, left→s1, down→s4
- s3: left→s3, down→s3, up→s1, right→s4
- s4: right→s4, down→s4, up→s2, left→s3

---

**Value-iteration calculation formulae**  
$V_{k+1}(s) = \max_{a \in A}\,[\,R(s) + \gamma\, V_k(s')\,]$

where ,

- $V_{k+1}(s)$: the **updated** value of $s$ after one more backup (the next iteration).
- $R(s)$: the **immediate reward** for being in state $s$ . Here $R(s_1)=5$, $R(s_2)=10$, $R(s_3)=1$, $R(s_4)=2$).
- $\gamma$: the **discount factor**, that says how much we value **future** rewards relative to **now**.

---

## Iteration 1

We start with all zeros, so the “future” term $\gamma V_0(s')$ is zero.

**Result:** each state’s updated value equals its **immediate reward**.

### 1) Initial value function $V_0$

- $V_0(s_1)=0,\ V_0(s_2)=0,\ V_0(s_3)=0,\ V_0(s_4)=0$

### 2) **Policy evaluation calculations** (explicit $Q_0$ from $V_0$)

- **State $s_1$** (neighbors: up/left→$s_1$, right→$s_2$, down→$s_3$)  
  $Q_0(s_1,\text{up}) = 5 + 0.9\cdot V_0(s_1) = 5 + 0 = 5$  
  $Q_0(s_1,\text{right}) = 5 + 0.9\cdot V_0(s_2) = 5 + 0 = 5$  
  $Q_0(s_1,\text{down}) = 5 + 0.9\cdot V_0(s_3) = 5 + 0 = 5$  
  $Q_0(s_1,\text{left}) = 5 + 0.9\cdot V_0(s_1) = 5 + 0 = 5$

- **State $s_2$** (up/right→$s_2$, left→$s_1$, down→$s_4$)  
  $Q_0(s_2,\text{up}) = 10 + 0.9\cdot 0 = 10$  
  $Q_0(s_2,\text{right}) = 10 + 0.9\cdot 0 = 10$  
  $Q_0(s_2,\text{left}) = 10 + 0.9\cdot 0 = 10$  
  $Q_0(s_2,\text{down}) = 10 + 0.9\cdot 0 = 10$

- **State $s_3$** (left/down→$s_3$, up→$s_1$, right→$s_4$)  
  $Q_0(s_3,\text{up}) = 1 + 0.9\cdot 0 = 1$  
  $Q_0(s_3,\text{right}) = 1 + 0.9\cdot 0 = 1$  
  $Q_0(s_3,\text{down}) = 1 + 0.9\cdot 0 = 1$  
  $Q_0(s_3,\text{left}) = 1 + 0.9\cdot 0 = 1$

- **State $s_4$** (right/down→$s_4$, up→$s_2$, left→$s_3$)  
  $Q_0(s_4,\text{up}) = 2 + 0.9\cdot 0 = 2$  
  $Q_0(s_4,\text{right}) = 2 + 0.9\cdot 0 = 2$  
  $Q_0(s_4,\text{down}) = 2 + 0.9\cdot 0 = 2$  
  $Q_0(s_4,\text{left}) = 2 + 0.9\cdot 0 = 2$

**Compact $Q_0$ table (same result):**

| State |  up | right | down | left |
| ----: | --: | ----: | ---: | ---: |
|    s1 |   5 |     5 |    5 |    5 |
|    s2 |  10 |    10 |   10 |   10 |
|    s3 |   1 |     1 |    1 |    1 |
|    s4 |   2 |     2 |    2 |    2 |

### 3) Policy improvement and value update

- Greedy action per state: all actions tie; choose a consistent tie-break (e.g., **up**).
- Update values: $V_1(s)=\max_a Q_0(s,a)$

| State | $V_1(s)$ |
| ----: | -------: |
|    s1 |        5 |
|    s2 |       10 |
|    s3 |        1 |
|    s4 |        2 |

---

## Iteration 2

Now $V_1$ captures which neighbors are promising. When we compute $Q_1(s,a)=R(s)+\gamma V_1(s')$, states that can move into (or stay in) high-value neighbors (like $s_2$) get larger values. The greedy policy begins to point **toward $s_2$**.

### Policy evaluation (compute $Q_1(s,a)$ from $V_1$)

- **s1**  
  up→$s_1$: $5 + 0.9\cdot 5 = 9.5$  
  right→$s_2$: $5 + 0.9\cdot 10 = 14.0$  
  down→$s_3$: $5 + 0.9\cdot 1 = 5.9$  
  left→$s_1$: $5 + 0.9\cdot 5 = 9.5$

- **s2**  
  up→$s_2$: $10 + 0.9\cdot 10 = 19.0$  
  right→$s_2$: $10 + 0.9\cdot 10 = 19.0$  
  left→$s_1$: $10 + 0.9\cdot 5 = 14.5$  
  down→$s_4$: $10 + 0.9\cdot 2 = 11.8$

- **s3**  
  up→$s_1$: $1 + 0.9\cdot 5 = 5.5$  
  right→$s_4$: $1 + 0.9\cdot 2 = 2.8$  
  down→$s_3$: $1 + 0.9\cdot 1 = 1.9$  
  left→$s_3$: $1 + 0.9\cdot 1 = 1.9$

- **s4**  
  up→$s_2$: $2 + 0.9\cdot 10 = 11.0$  
  left→$s_3$: $2 + 0.9\cdot 1 = 2.9$  
  right→$s_4$: $2 + 0.9\cdot 2 = 3.8$  
  down→$s_4$: $2 + 0.9\cdot 2 = 3.8$

**Compact $Q_1$ table:**

| State |   up | right | down | left |
| ----: | ---: | ----: | ---: | ---: |
|    s1 |  9.5 |  14.0 |  5.9 |  9.5 |
|    s2 | 19.0 |  19.0 | 11.8 | 14.5 |
|    s3 |  5.5 |   2.8 |  1.9 |  1.9 |
|    s4 | 11.0 |   3.8 |  3.8 |  2.9 |

### Policy improvement (greedy w.r.t. $Q_1$)

- $\pi_2(s_1)=$ **right** (14.0)
- $\pi_2(s_2)=$ **up** (tie with right at 19.0; choose up consistently)
- $\pi_2(s_3)=$ **up** (5.5)
- $\pi_2(s_4)=$ **up** (11.0)

### Value update (compute $V_2$)

- $V_2(s_1)=14.0,\ V_2(s_2)=19.0,\ V_2(s_3)=5.5,\ V_2(s_4)=11.0$

**Table of $V_2$:**

| State | $V_2(s)$ |
| ----: | -------: |
|    s1 |     14.0 |
|    s2 |     19.0 |
|    s3 |      5.5 |
|    s4 |     11.0 |
