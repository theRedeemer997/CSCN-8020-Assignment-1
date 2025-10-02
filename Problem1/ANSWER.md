# Problem 1 — Pick-and-Place Robot

**Question:**  
Pick-and-Place Robot: Consider using reinforcement learning to control the motion of a robot arm  
in a repetitive pick-and-place task. If we want to learn movements that are fast and smooth, the  
learning agent will have to control the motors directly and obtain feedback about the current positions  
and velocities of the mechanical linkages.  
**Design the reinforcement learning problem as an MDP, define states, actions, rewards with reasoning.**

---

# Answer

We design the pick-and-place robot task as a Markov Decision Process (MDP) with **States (S) , Actions (A) , Rewards (R) , Transitions (P) , and a Discount factor (γ) .**

---

## States (S)

The different states required here to capture the key steps of a pick-and-place task are:

- Idle (arm waiting, gripper open)
- Moving toward the object
- Holding the object
- Moving toward the target bin
- Placing the object

---

## Actions (A)

The different moves needed to complete the pick-and-place task are:

- Move the arm (up, down, left, right)
- Open the gripper
- Close the gripper

---

## Rewards (R) — points for good or bad behavior

Rewards are given to the robot for the good behaviour as well as penalties are also given to them for the bad behaviour. Postive rewards encourage the robot to finish the task while the penalties teach it to be fast, smooth and safe.

The different rewards and penalties are:

- **+10** → when it successfully places the object in the bin
- **+5** → when it successfully picks up the object
- **-1** → for every step (so it learns to be quick)
- **-5** → if it crashes, collides, or makes a mistake

---

## Transitions (P) — how states change

This shows how the robot progresses through each task.

- δ(Idle, move_to_object) → Moving_to_object
- δ(Moving_to_object, close_gripper) → Holding_object
- δ(Holding_object, move_to_bin) → Moving_to_target
- δ(Moving_to_target, open_gripper) → Placing_object
- δ(Placing_object, release) → Idle

---

## Discount factor (γ)

We use a discount factor such as **0.9**.

This makes the robot value **faster rewards** more than future ones and also pushes the robot to complete the task quickly instead of wasting time.
