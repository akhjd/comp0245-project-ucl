# Proposal: the learned arm (torques)

Status: proposal for Akhilesh to decide. Nothing built yet.

## What the brief needs
The model must output `torque_arm` (7 joint torques) from the robot's measurements (arm joint angles `q`,
speeds `qd`, base pose, target). A design where the model outputs a target pose and a separate PD makes the
torque would not meet that, so the model itself must produce torques.

## What the expert arm actually does (course code, `robotnik_arm_control.arm_controller`)
```
torque = kp*(q_des - q) + kd*(qd_des - qd) + g(q)        kp = 100, kd = 30
```
- `g(q)` = **gravity compensation**: the torque needed just to hold the arm up at angles `q`. Pure physics,
  from the robot model (pinocchio). It is usually the largest part of the torque.
- `q_des`, `qd_des` = a moving reference produced by differential IK toward the target. `q_des` is an
  **integrated** (remembered) quantity that leads the arm by up to 0.15 rad, i.e. up to 15 N m of hidden
  torque. It is not in the observation (LESSONS.md #5).
- The base is stationary while the arm reaches, so the arm problem is "target relative to the robot ->
  torques" and does not depend on where the robot is in the world.

## Recommended design: gravity term from physics + learned remainder
```
torque = g(q)  +  network(inputs)
```
- `g(q)` is computed from the robot model, exactly as the expert does. It is physics, not an expert internal:
  it only needs the measured joint angles. Course link: W1-2 "The manipulator linear dynamic regressor"
  (robot dynamics are known up to parameters). The brief's requirement is still met: the model outputs the
  full torque; it has a known physical part and a learned part.
- The network learns only the "reaching" part `torque - g(q)` (= the PD toward the moving reference).
  That is smaller, smoother and much easier to learn than gravity mixed in.
- **Baseline / ablation:** the same network learning the full torque directly (no `g(q)`). Report both; we
  expect the direct version to sag or drift.

### Inputs (all from sensors plus forward kinematics)
| Input | Size | Why |
|---|---|---|
| joint angles: sin and cos for the 4 continuous joints, raw angle for the 3 limited ones | 11 | angles wrap; same reason as heading sin/cos |
| joint speeds `qd` | 7 | needed for damping (the `kd` term); without it the arm oscillates |
| target relative to the hand, in the robot frame (FK of `q`) | 3 | "how far and which way still to go"; the main signal |
| target relative to the shoulder, in the robot frame | 3 | where in the reach zone the target is |
| **total** | **24** | |

### Output
7 numbers: `torque - g(q)` (N m), standardised like the wheels.

### Data
- Only the arm-reaching phase (after the base has parked): about 2-3 s per episode, the same course arm expert
  in every dataset, so dataset-v3 (80) + obst-v2b (80) + obst-v3-extra (160) can all be used.
- Keep every 10th row as for the base.

### Copycat risk
`qd` is needed (damping), but it is also the kind of input that caused the base copycat problem. Plan: train
with `qd`, check closed loop; if the arm stalls or keeps moving the way it is already moving, try a short
history of `q` instead, and/or DAgger (see below).

### Evaluation (closed loop, validation scenarios only)
1. **Arm alone (hybrid, labelled as such):** expert base parks, learned arm reaches. Final hand error, time to
   within 1 cm, torque clipping, oscillation (torque sign flips per second), joint-limit hits.
2. **Full learned robot:** learned base + learned arm. Success = hand within tolerance, no contact.
3. **Efficiency:** parameters, inference time per step (must be well under 1 ms).
4. **Ablations:** with vs without the physics gravity term; with vs without `qd`.

### DAgger for the arm (if needed)
Harder than for the base, because the arm expert has memory (`q_des`). To label a state the policy visited,
the expert needs a reference; the usual approximation is to restart it at `q_des = q` (no lead). That is an
approximation and would be stated as such.

## Effort
About half a day: data extraction (1 h), training script reuse (comp0245/training/train_base_torch.py pattern, 1 h), arm policy
in the evaluation (1-2 h), runs and analysis (1-2 h). Akhilesh could write the training part, guided, as before.

## Decisions for Akhilesh
1. Physics gravity term + learned remainder (recommended), or learn the full torque directly?
2. Write the training script yourself (guided), or let Claude build all of it?
