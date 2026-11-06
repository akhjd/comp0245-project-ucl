# Lessons learned

Every real problem hit in this project so far, in the order it happened: what we saw, what caused it, how it
was found, the fix, and the general lesson. Includes Claude's own mistakes, marked **(Claude's mistake)**.
Sources: AGENT_LOG.md (full detail and numbers) and runs/ / archive/runs/ (the evidence).

---

## Part 1: Learning-side problems (behaviour cloning)

### 1. Low offline error, but the robot drives off (first models, 26-27 Oct)
- **Seen:** ridge, MLP and random forest all had small validation errors on recorded data, yet in the
  simulator the base drove away (final errors of tens to hundreds of metres; mlp-v3 even blew up the physics
  with saturated arm torques).
- **Cause:** *compounding error / covariate shift.* A model trained on the expert's states makes a small
  mistake, lands in a state the expert never visited, makes a bigger mistake there, and so on.
- **Found by:** running every model in closed loop (letting it drive) instead of trusting the offline score.
- **Lesson:** offline error measures copying on the expert's own path. Only closed-loop tests measure
  driving. Always report both. (Ross et al. 2011, DAgger.)

### 2. Too little variety in the data (6 episodes, 4 targets)
- **Seen:** the first dataset had 4 training targets, all from the same start.
- **Cause:** with 4 targets a model cannot learn "target -> action"; base position and time effectively told it
  which episode it was in.
- **Fix:** random targets (dataset-v3, 60/10/10), then random start poses and obstacles (obst-v2b).
- **Lesson:** count *distinct situations*, not rows. 520,000 rows from 60 episodes is still 60 situations.

### 3. Inputs blown up by scaling
- **Seen:** some inputs (e.g. an arm joint that barely moves) had a tiny training spread, so StandardScaler
  turned small changes in closed loop into z-scores in the thousands.
- **Lesson:** check each input's spread before standardising; near-constant inputs are dangerous.

### 4. World-frame inputs
- **Seen:** inputs were world coordinates, so the same situation at a different place looked different.
- **Fix (Akhilesh):** robot-frame features (`target_in_base_frame`, `base_inputs`): forward/left distance,
  heading error as sin/cos, velocities rotated into the robot frame.
- **Lesson:** describe the situation from the robot's point of view; then one example teaches every location.

### 5. Hidden expert state
- **Seen:** the course expert has internal memory (phase, a moving arm reference) that is not in the
  observation, so the same observation can need different actions.
- **Decision (Akhilesh):** policies only get sensor values; expert internals may be logged for diagnosis,
  never fed to a model. Our obstacle expert was made memoryless and sensor-only partly for this reason.
- **Lesson:** a student can only copy what it can see.

### 6. The base model stops short (base-mlp-v1)
- **Seen:** 9 of 10 runs stopped 4-16 cm before the parking spot.
- **Cause (likely):** the expert switches sharply from "creep" to "stop"; a ReLU network can only make that
  jump out of slopes, so it slows early (regression averaging near a discontinuity).
- **Lesson:** sharp switches in the expert are hard to learn; expect smoothing near them.

### 7. Copycat / velocity shortcut (obst-base-torch-v1, 28 Oct)
- **Seen:** the PyTorch lidar policy had the *better* offline error (0.193 vs 0.567 rad/s) but parked 0/10,
  touched obstacles 4/10, barely turned (0.03 rad/s with a 75 degree heading error), and sat still when the
  target was behind it.
- **Cause:** its inputs included its own current speed and turn rate. In expert data "turning now" is almost
  always followed by "keep turning", so the network learned to copy its current motion. In closed loop:
  not turning -> never starts turning.
- **Found by:** printing turn rate vs heading error over time in failed episodes; then an ablation without the
  3 velocity inputs: 0 contacts, 7/10 parked within arm reach.
- **Lesson:** offline error and closed-loop success can move in opposite directions. Test causes with an
  ablation (change one thing). (Codex predicted this earlier; see AGENT_LOG.md.)

### 8. The model meets states the expert never showed it
- **Seen:** even without velocity inputs, runs that start facing away from the target fail.
- **Cause:** the expert always turns early, so the data has almost no "far away, facing wrong, stopped" states.
- **Fix in progress:** DAgger: the model drives, the expert labels what it would do in every visited state.
  Possible here only because our expert is memoryless and sensor-only.

### 9. Early stopping
- **Seen:** best validation loss at epoch 19, but epoch 30's weights were saved.
- **Fix:** comp0245/training/train_base_torch.py now keeps the best-validation epoch.

---

## Part 2: Simulator and physics problems

### 10. Control rate matters (Codex, 25 Oct)
- **Seen:** at 200 Hz control even the expert became unstable; at 1 kHz it succeeded.
- **Lesson:** keep the course's 1 kHz rate; check timing before blaming the controller.

### 11. Ice-like wheels **(Claude's mistake: should have measured first)**
- **Seen:** the obstacle expert kept sliding, spiralling and running away at 1 m/s during curves.
- **Cause:** the course sets the wheels' sideways grip to 1% (`anisotropicFriction` 0.01). Sustained curves
  build a sideways slide that grows. The course expert avoids it by turn, drive straight, turn.
- **What went wrong in the process:** Claude spent several iterations tuning the controller (braking rules,
  slide damping, gentler curves) before measuring the physics in isolation. One 20 s test drive with a
  fixed command showed the cause immediately.
- **Fix (Akhilesh's decision):** sideways grip 0.1 for obstacle runs only (slide stays below 0.11 m/s); the
  course task is unchanged. Stated openly as a physics change.
- **Lesson:** when a controller misbehaves, first test the plant with simple fixed commands. Measure before
  tuning.

### 12. Spinning on the spot slides the base
- **Seen:** a half turn on the spot moves the base 0.3-0.4 m sideways.
- **Effect:** wiggly starts; the "arrived" zone had to be wider than that slide or turning pushed the robot
  back out.
- **Lesson:** a model of the robot as a point that turns in place is wrong for skid-steer; plan for it.

---

## Part 3: Obstacle expert design problems

### 13. Flipping left/right forever
- **Seen:** with an obstacle nearly dead ahead the expert dithered: left, right, left.
- **Cause:** with one lidar ray every 10 degrees, the obstacle's apparent centre shifts as the robot turns, so
  any fixed "pass on the far side" rule has a boundary the robot can sit on.
- **Fix:** near a tie, pass on the side the robot already faces; turning then reinforces the choice.
- **Lesson:** a memoryless rule needs hysteresis built into its geometry, or it chatters at boundaries.

### 14. Late, wide swerves
- **Seen:** the expert drove straight at an obstacle, noticed it at 0.5 m, swerved about 90 degrees and swung back
  (path 1.48x the straight line, a full turn of total rotation).
- **Fix:** only the part of the motion heading INTO an obstacle is turned sideways, from 1 m away. Path 1.27x,
  turning halved, clearance 0.14 -> 0.38 m.
- **Lesson:** measure quality (path ratio, turning, clearance), not just success; a 100% successful expert
  can still be a bad teacher.

### 15. A hole introduced by fix 14 **(Claude's mistake, caught by its own checks)**
- **Seen:** after fix 14, a few training runs passed 1.5 cm from an obstacle while turning from the start.
- **Cause:** the new rule only checked the *goal direction*, not the direction the robot was actually facing.
- **Found by:** checking minimum clearance over the full 80-episode dataset, not only the 24 trial runs.
- **Fix:** use whichever of the two points more into the obstacle. Remaining limit: the robot is modelled as a
  circle, so a swinging corner can still pass within a few cm (no contact seen).
- **Lesson:** after a fix, re-check on more and different cases; small test sets hide edge cases.

### 16. Inverted sign in the side rule **(Claude's mistake)**
- **Seen:** all 16 trial runs timed out.
- **Cause:** `side > deadband` instead of `side < deadband`.
- **Lesson:** unit-check a new rule on one hand-made case before launching a batch.

### 17. The arm reached after a navigation timeout
- **Seen:** when the base gave up, the arm still reached for a target metres away and swung into an obstacle.
- **Fix:** no reach after a timeout.

### 18. Parking at the edge of the tolerance
- **Seen:** the base stopped 14 cm from its spot (tolerance 15 cm).
- **Fix:** inside the zone, face the target and creep along the robot-target line until within 5 cm.

### 19. A "success" that wasn't
- **Seen:** one episode scored success with the arm never moving: the folded hand was driven onto a low target.
- **Fixed as a side effect** of parking further back (reach zone).
- **Lesson:** read the episodes behind a success count.

---

## Part 4: Tooling and process problems

### 20. sklearn and Ctrl-C
- **Seen:** stopping training with Ctrl-C silently saved a half-trained model (mlp-v3-interrupted-9ep).
- **Lesson:** sklearn catches the interrupt; name and archive such runs so they are never mistaken for real ones.

### 21. Viewer canvas kept growing (found by Codex)
- **Cause:** on high-DPI screens the canvas height was read back after being scaled. Fixed by storing the
  logical height.

### 22. Hidden simulator output **(Claude's mistake)**
- **Seen:** a filtered log showed nothing for the first episode.
- **Cause:** pybullet prints without a newline, so an anchored grep dropped the line.

### 23. Parallel runs into one folder
- **Seen:** two collection processes writing to the same output folder; one refused to start.
- **Lesson:** one output folder per process (or one process), then combine.

### 24. Shapes in PyTorch layers (learning moment)
- **Seen:** `mat1 and mat2 shapes cannot be multiplied (256x64 and 4x64)`.
- **Cause:** a layer's input size must equal the previous layer's output size; `nn.Linear(in, out)`.

### 25. Disk space
- **Seen:** at 1,000 rows per second a 40 s episode is about 18 MB; DAgger would have used about 8 GB.
- **Fix:** DAgger runs save every 10th row (training reads every 10th row anyway).

### 26. Measurement bugs in the robustness suite (found by Codex) **(Claude's mistakes)**
- **Seen (Codex review, 28 Oct):** (a) "parked" was checked at one saved moment, on linear speed only: a robot
  spinning on the spot at 1 rad/s passed strict. (b) The delay test applied the first command at once and repeated
  it until the queue filled, so the start was not delayed, and the trajectory recorded the requested command, not
  the one the motors got. (c) The expert reference also reached with its arm, held it with gravity compensation
  and ended the episode early; the learned bases held the arm with a plain PD for 40 s. Different exposure, so
  not a fair comparison. (d) Models trained on a GPU would not load on the CPU.
- **Fix (29 Oct):** strict/loose now have to hold for the last 2 s (distance, facing, speed and yaw rate); wheels
  get zero until the first delayed command arrives, and `applied_actions` is saved; a new `expert_hold`
  reference uses the expert's wheels with the same arm hold and episode length; `map_location="cpu"` plus an
  input-name check. Old runs re-scored without re-simulating (`comp0245/evaluation/rescore.py`, written to `rescored.json`; originals kept).
- **What it changed:** the clean-trained no-velocity model had looked perfectly robust to lidar noise (30/30
  loose); with the 2 s check it is 19/30 at the strongest noise, because it keeps twitching after parking.
- **Lesson:** a metric that samples one moment can be passed by behaviour you would never accept. A reference
  must face exactly the same conditions as the thing compared with it. Have someone else review the test bench.

### 27. Two good modules, worse together (interface shift)
- **Seen (29 Oct):** the arm after DAgger reached 40/40 when the expert parked the base, but only 23/40 when the
  learned base parked it, although the learned base itself parks within the loose limits almost every time.
- **Cause:** the arm was only ever trained from the expert's parkings, where the target is never more than 1 cm to
  the side. The learned base leaves it up to 5 cm to the side. Small for the base's own score, but a situation
  the arm had never seen, so it hovered near the target and drifted, as before DAgger.
- **Lesson:** test the pieces separately (to know whose fault a failure is) AND together; the output of one
  learned module is the input distribution of the next. Train the second module on what the first one really
  produces (here: arm DAgger with the learned base driving).

---

## General rules that came out of this
1. Judge a policy by closed-loop driving, never by offline error alone.
2. Change one thing at a time and keep the old run (ablations are the evidence).
3. Measure the plant before tuning the controller.
4. Check quality metrics, not only success.
5. Re-verify fixes on more cases than the ones used to make them.
6. Keep test data untouched until the end.
7. Test the test bench: a fair reference, a metric that cannot be gamed, and the disturbance you think you apply.
8. When learned parts are chained, train each part on what the part before it really produces.
