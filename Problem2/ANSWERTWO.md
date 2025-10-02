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

**Backup used (Value Iteration)**
\[
V*{k+1}(s) \leftarrow \max*{a \in A}\big[\,R(s) + \gamma\, V_k(s')\big]
\]
(Policy improvement is the greedy action achieving the max.)

---

## Iteration 1

### 1) Initial value function \(V_0\)

\[
V_0(s_1)=0,\quad V_0(s_2)=0,\quad V_0(s_3)=0,\quad V_0(s_4)=0
\]

### 2) Perform value-function updates (backup from \(V_0\))

Because \(V_0(\cdot)=0\), each backup reduces to the immediate reward:

- \(V_1(s_1)=\max_a[5 + 0.9 \cdot 0] = 5\)
- \(V_1(s_2)=\max_a[10 + 0.9 \cdot 0] = 10\)
- \(V_1(s_3)=\max_a[1 + 0.9 \cdot 0] = 1\)
- \(V_1(s_4)=\max_a[2 + 0.9 \cdot 0] = 2\)

### 3) Updated value function \(V_1\)

| State | \(V_1(s)\) |
| ----: | ---------: |
|    s1 |          5 |
|    s2 |         10 |
|    s3 |          1 |
|    s4 |          2 |

_(Policy improvement note: all actions tie here since the lookahead term was 0; keep any consistent tie-break, e.g., **up**.)_

---

## Iteration 2

### Value function after the second iteration \(V_2\)

Now use \(V_1\) in the backup \(V_2(s)=\max_a [R(s)+0.9\,V_1(s')]\):

- s1: best next is s2 (value 10) ⇒ \(V_2(s_1)=5+0.9\times10=14\)
- s2: best next is s2 (value 10) ⇒ \(V_2(s_2)=10+0.9\times10=19\)
- s3: best next is s1 (value 5) ⇒ \(V_2(s_3)=1+0.9\times5=5.5\)
- s4: best next is s2 (value 10) ⇒ \(V_2(s_4)=2+0.9\times10=11\)

**Final table (after Iteration 2):**
| State | \(V_2(s)\) |
|------:|-----------:|
| s1 | 14.0 |
| s2 | 19.0 |
| s3 | 5.5 |
| s4 | 11.0 |

_(Policy improvement note: greedy actions w.r.t. \(V_2\): s1→right, s2→up (or right, tie), s3→up, s4→up.)_
