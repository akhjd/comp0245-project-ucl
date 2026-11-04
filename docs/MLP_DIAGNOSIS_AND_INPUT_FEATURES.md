# MLP diagnosis and proposed input features

Written 2 Nov 2026. This covers why the first MLP cannot drive, what the imitation learning survey suggests, and the proposed new lidar and dot inputs.

**Status:** the lidar and dot features below are a proposal and are not implemented. Removing velocity needs the user's approval. No model has been trained on them.

## Summary

- The expert solves 98.5% of runs, but the first MLP (`mlp1`) has not succeeded once in 170 driving runs.
- Its validation R² of about 0.95 is misleading, because it mostly follows the robot's current velocity.
- It largely ignores the dot positions, so it has no real goal.
- This best fits the copycat problem, but that is inferred from measurements and not yet proven. A shared bug in training and driving is not ruled out.
- The small gap between train and validation loss is not the main cause.
- The proposal is to replace 180 raw lidar values with 30 meaningful features, express dots as distance and heading, and test removing velocity.

## Current training

| Item | `mlp1` |
|---|---|
| Training data | 1,461 successful expert runs, 5.15 million rows at 100 rows per second |
| Correct answer | The 4 wheel speeds the expert commanded on that row |
| Network | 3 layers of 256 units, 183,812 parameters |
| Method | Shuffled rows, batches of 1,024, Adam, mean squared error |
| Learning rate | Starts at 0.001, halves after 4 flat epochs |
| Epoch limit | 60 |
| Epochs run | 17, stopped after 12 with no improvement |
| Model used | Epoch 5, about 25,000 updates |

Training scores only how well the wheel commands match the expert's. It never sees whether the robot reached a dot.

### The 199 inputs

| Input | Calculation | Values |
|---|---|---|
| Lidar history | 36 rays x 5 frames (now, and 0.25, 0.5, 0.75 and 1 s ago) | 180 |
| Pose | x, y, sin(yaw), cos(yaw) | 4 |
| Velocity | Forward speed, sideways speed, yaw rate | 3 |
| Dots | 3 slots x 4 values (forward, left, height difference, still-there flag) | 12 |
| **Total** | | **199** |

A removed or unused dot slot is all zeros.

## Results

Expert runs count as failed if they touched anything, even after reaching every dot. Failed runs were left out of the training data.

| Case | Expert, training scenarios | `mlp1`, training scenarios | `mlp1`, validation scenarios |
|---|---|---|---|
| 1: easy room, no obstacles | 135/135 | 0/30 | 0/8 |
| 2: easy room, obstacles | 356/360 | 0/30 | 0/20 |
| 3: hard room, no obstacles | 270/270 | 0/28 | 0/15 |
| 4: hard room, obstacles | 441/450 | Not reached | 0/24 |
| 5: hard room, obstacles, 3 dots | 259/270 | Not reached | 0/15 |
| **Total** | **1,461/1,485** | **0/88** | **0/82** |

- Dots touched by `mlp1`: 0/88 on training scenarios and 1/112 on validation scenarios.
- The validation runs ended with 51 timeouts, 19 crashes (2 s of contact) and 12 stalls (20 s without moving 5 cm).
- 18 of the 19 crashes involved walls or furniture, and 1 involved a moving obstacle.
- The expert succeeded in 1,625 of 1,650 runs across all splits.

DAgger was stopped by the user in round 1 with 88 of 150 driving runs saved, so no DAgger model exists.

## Why the MLP fails

Checks on validation rows:

| Check | Result |
|---|---|
| Normal model | R² 0.947 |
| Velocity hidden | R² -0.15 |
| Velocity alone, linear fit | R² 0.898 |
| All lidar hidden | R² 0.91 |
| Dots hidden | R² 0.947, no change |
| Pose hidden | R² 0.948, no change |
| Robot still while the expert commands motion | Expert 2.26 rad/s, MLP 0.06 rad/s |

Every run starts with the robot still, but only about 0.03% of validation rows have the robot still while the expert commands motion. In nearly every other row, velocity largely gives away the answer. So the MLP scores well on expert data, but from a standstill it predicts almost no motion and then creeps around without a goal. This can happen even in an empty room.

### Is this overfitting?

Only slightly, and it does not explain the failure.

- From epoch 5 to 17, train loss fell from 0.032 to 0.015 while validation loss rose from 0.051 to 0.060.
- That rise is small and noisy, and the `OVERFITTING` verdict comes from a simple automatic rule.
- True overfitting to the task would mean driving well on training scenarios and badly on new ones. `mlp1` fails on both.

A better description is a slight offline train/validation gap, with the main failure being velocity copying and poor recovery away from the expert's path.

### Bug checks

| Possible bug | Status |
|---|---|
| Driving inputs differ from training inputs | Ruled out: commands match on saved rows to within 0.000005 rad/s |
| Wrong columns for yaw, speed or yaw rate | Code read against the column names: they match |
| Current dot marked as gone | Ruled out by the dataset check |
| A mistake shared by training and driving | Not ruled out: both use the same input code |
| Arm logic blocking success while the MLP drives | Unlikely to explain most failures, not separately tested |

## Survey: what is relevant

Paper: Zare et al., *A Survey of Imitation Learning: Algorithms, Recent Developments, and Challenges*, [arXiv:2309.02473](https://arxiv.org/abs/2309.02473). It reviews methods and does not test them on a task like ours. Reference numbers below are the survey's.

| Rank | Technique | Relevance | Effort |
|---|---|---|---|
| 1 | Causal confusion and the copycat problem (Sec. II; [34], [35], [36]) | Closely matches the measurements. The paper describes copying previous actions; treating velocity as the same shortcut is our interpretation | Low for a simple version |
| 2 | DAgger (Sec. II; [14]) | Already built. Addresses covariate shift, and our expert is code, so human workload does not apply | Done |
| 3 | Expert handover: SafeDAgger, HG-DAgger, LazyDAgger ([23], [21], [24]) | Could get runs past the start, so DAgger reaches doorways, parking and later dots | Fairly small |
| 4 | Synthetic perturbations and extra losses ([31]) | Could teach recovery without the MLP driving | Medium |
| 5 | Implicit behavioural cloning ([38]) | May help when the expert's choices would average into a poor command | High, and slower to run |

Implicit cloning can still use an MLP, but the MLP takes the inputs plus a candidate wheel command and outputs a score. Driving tries many candidates and picks the lowest score. It would not fix velocity copying by itself. Check that the assignment allows it.

Minor or later: ensemble disagreement ([27]), ThriftyDAgger ([20]), error detection ([32]), varied dynamics ([104]), and more varied data (Sec. VII).

Not relevant: inverse RL and adversarial imitation (expensive, and we can already ask the expert directly), imitation from observation (we have the actions), imperfect demonstrations (we kept clean runs), and domain differences (same robot and simulator).

Useful for the report: covariate shift, causal confusion, and the Fig. 2 grouping of methods. The survey does not discuss input noise or mirroring as fixes.

## Proposed inputs

The idea is the same as converting xyz coordinates to distance and heading error: give the MLP the meaning directly instead of making it work this out from raw numbers. The features can be calculated from the saved lidar, so no new data collection is needed. They use only lidar and dot positions, so there is still no A* and no map.

### Lidar: 180 values to 30

| Feature | Calculation | Values | What it tells the MLP |
|---|---|---|---|
| Nearest obstacle per sector | Shortest reading in each 30 degree sector, now | 12 | Where it is blocked and where it is open |
| Change per sector | Sector distance now minus 0.5 s ago | 12 | What is getting closer, which helps with movers |
| Closest obstacle | Distance, plus sin and cos of its bearing | 3 | The most urgent thing to avoid |
| Free space towards each dot | Lidar distance in the direction of each dot slot | 3 | Whether the way to that dot looks open |

### Dots: 12 values to 15

Each of the 3 slots becomes distance, sin and cos of heading error, height difference, and the still-there flag. Hiding the dots currently changes almost nothing, so making them easier to use is worth testing.

### Risks

- **Lidar is not the main problem:** hiding all lidar only dropped R² from 0.947 to 0.91, while hiding velocity dropped it to -0.15. The MLP could still rely on velocity after this change.
- **Lidar change can leak speed:** distances change when the robot moves, so these 12 values could bring the shortcut back even with velocity removed. If that happens, drop them.
- **Less detail:** a sector keeps only its nearest obstacle, so a narrow doorway may look blocked. The 10 degree rays may be needed to find doorways.

## Test plan

Change one thing at a time on the same data, and score each model by driving, not by R².

| Model | Lidar and dots | Velocity |
|---|---|---|
| A | Current raw inputs | Removed |
| B | New features | Kept |
| C | New features | Removed |

- If only A and C drive, velocity was the problem.
- If B drives too, the new features were enough on their own.
- Start with case 1 only. If nothing drives even there, look for a bug before tuning anything else.
- Rough cost per model: 15-30 minutes of training and about an hour of driving tests.
- Compare against `mlp1`: 0/82 successes and 1/112 dots.

Velocity stays in the main model unless these results justify dropping it, because it is one of the user's chosen inputs.

## Other experiments and suggested order

Input noise and left-right mirroring are in `train_mlp.py` as `mlp2`. The mirror passed its simulator checks. They mainly target generalisation, which is not clearly the problem, and they cannot add the missing standstill examples, so they may not change driving much. The `mlp2` run is optional.

1. Add the new features and a velocity option, including a case 1 filter.
2. Run models A, B and C on case 1.
3. Take the best inputs to all five cases, and run the `mlp2` test if still wanted.
4. Continue DAgger from the 88 saved runs with the better model, adding expert handover if runs still stall early.
