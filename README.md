# UCL COMP0245 — Machine Learning for Robotics

Behaviour cloning for a mobile manipulator: learn commands from expert trajectories to reach Cartesian end-effector targets and evaluate generalisation to new targets and initial conditions.

The project extends the course task in two stages: first an open floor with still cylinders and a 36-ray lidar, then walled rooms with **moving** cylinders (and, as an extension, Julia's floor plans). Report: four systems, in the order they were built (numbers below are from the logged runs; every run folder is listed in [docs/DATASETS.md](docs/DATASETS.md)):

1. **Open-floor robot** (this section and the next): learned base + learned arm among still cylinders, made robust to lidar noise, pushes and delays.
2. **Room model** ([weights/base_rooms.pt](weights/base_rooms.pt)): base for walled rooms with moving cylinders; the final model, tested once on 80 rooms.
3. **Room arm** ([weights/arm_rooms.pt](weights/arm_rooms.pt)): the arm retrained to reach from where the room base parks.
4. **Combined model** (6 Nov, not adopted): one base for our rooms and Julia's floor plans, with a faster 0.8 m/s teacher and a 144-ray 10 m lidar.

The open-floor system has two learned modules, both trained by behaviour cloning plus DAgger from the course expert (base: a sensor-only steer-around expert; arm: the course arm controller):

- **Base policy** ([weights/base_final.pt](weights/base_final.pt)): 4 wheel speeds from 42 inputs (target in the robot's frame + 36 lidar ranges, no velocity inputs), MLP 64-64. Noise-robust DAgger with hard layouts mixed in (`dagger.py`).
- **Arm policy** ([weights/arm_final.pt](weights/arm_final.pt)): 7 joint torques = gravity torque (pinocchio, physics) + MLP 64-64 on 24 arm inputs (`models.arm_inputs`). DAgger with a shadow copy of the arm expert as teacher, first with expert parking, then with the learned base parking (`dagger_arm.py`).

Recorded datasets and run outputs are not bundled (see [data/README.md](data/README.md)); the final weight files are (30-60 kB each, [weights/README.md](weights/README.md)). The launch scripts behind the logged experiments are in [experiments/](experiments/).

## Open-floor system: results (validation)

All numbers are on validation scenarios; the test splits are reserved for one final evaluation and have not been used. Success = the course rule (hand within 1 cm of the target for 0.25 s) with no collision.

| Setup (40 fresh scenarios, `scenarios_obstacles_val_d.json`) | Success | Collisions | Median hand error |
|---|---|---|---|
| Fully learned: learned base parks, learned arm reaches | 37/40 | 0 | 6.5 mm |
| Expert wheels + learned arm | 40/40 | 0 | 6.6 mm |
| Full-task expert | 40/40 | 0 | 6.2 mm |

Base policy alone (arm held still, 30 fresh scenarios each; parked within 15 cm of the stand-off distance and 0.2 rad of facing the target, held still for the last 2 s, no collision): normal layouts 30/30, heavy lidar noise (0.3 m, 30% dropouts) 30/30, 120 N sideways push 28/30, 400 ms wheel delay 28/30, harder unseen layouts 27/30. Details, every run and how each set was used: [docs/DATASETS.md](docs/DATASETS.md); problems found along the way: [docs/LESSONS.md](docs/LESSONS.md).

## Room model: walled rooms with moving cylinders

The base drives with the arm folded inside its outline and unfolds the arm once parked. Teacher: a privileged dynamic-window controller (`obstacle_env.WindowExpert`; A* waypoints in floor plans) that the sensor-only student can copy, unlike the earlier RRT teacher. Student: 3 lidar scans 0.5 s apart (72 rays, 4 m), a learned 1-D convolutional ray filter, a training-only threat head, a learned hold output; no velocity inputs; DAgger with crossing-mover scenes.

| 60 validation rooms (base only, 70 s) | Mover hits | Parked |
|---|---|---|
| First room base: RRT teacher, one scan | 45 | 4 |
| + no velocity, 3 scans, ray filter, threat head | 30 | 15 |
| + dynamic-window teacher | 23 | 21 |
| + arm folded while driving | 16 | 31 |
| + crossing-mover DAgger, hold output (**final room model**) | 12 | 37 |

**Final test** (80 rooms of `scenarios_rooms_test.json`, run once on 5 Nov): learned base 48/80 parked (18 mover hits); full learned robot 45/80 (every one of its 16 contact runs was a mover hit).

**Room arm** (`arm_rooms.pt`, 5 Nov, validation only): retrained from 1,000 expert reaches started where the room base parks, plus 2 arm DAgger rounds; reaches from 45 of 49 real parking spots (the previous arm: 25). The final test used `arm_final.pt`.

**Combined model** (`runs/dagger-overnight-combined`, 6 Nov, validation only): teacher `FastWindowExpert` at 0.8 m/s (parks 55/60 in our rooms with 1 mover hit, 55/60 in Julia's), lidar 144 rays / 10 m, 2,100 rooms of seven types, 3 DAgger rounds. Ours 39/60 parked with 15 mover hits; Julia's 34/60 (the room model: 19/60). Not adopted: the rule set before the run allowed at most 13 mover hits.

```bash
# The room robot (room base + room arm) on a scenario file, with the PyBullet viewer
"$ROBOENV_PYTHON" run_scene.py --scenarios scenarios_rooms_val.json --out runs/scenes --gui
```

## Demonstration and evaluation of the open-floor models

These need PyTorch in the RoboEnv environment (see Setup). Run from the repository root.

```bash
# One episode in the PyBullet viewer, real time: full | base | arm | expert
"$ROBOENV_PYTHON" demo.py full
"$ROBOENV_PYTHON" demo.py base --scenario validationd_005
"$ROBOENV_PYTHON" demo.py full --file scenarios_obstacles_hard2_gate.json --scenario validationhard2g_000

# Scored evaluations (results in runs/arm-eval-NAME and runs/robust-NAME; use new names)
"$ROBOENV_PYTHON" eval_arm.py --name full --arm weights/arm_final.pt --base weights/base_final.pt \
    --fresh 0 40 --no-held-out --fresh-file scenarios_obstacles_val_d.json
"$ROBOENV_PYTHON" eval_arm.py --name arm-only --arm weights/arm_final.pt \
    --fresh 0 40 --no-held-out --fresh-file scenarios_obstacles_val_d.json
"$ROBOENV_PYTHON" robustness.py --name base --models final=weights/base_final.pt

# Replay pages (top-down animation of every episode, opens in a browser)
"$ROBOENV_PYTHON" view_runs.py runs/arm-eval-full/fresh-normal
```

## How the open-floor models were made

Each stage's output folder is listed in [docs/DATASETS.md](docs/DATASETS.md); every script has `--help`.

1. Scenarios: `make_obstacle_scenarios.py` (normal layouts, and `--hard` for walls, blockers, tight gaps and crowded parking; only expert-solved hard layouts were kept).
2. Expert data: `collect_obstacles.py --scenarios FILE --out runs/NAME` for v2, v3_extra (base) and the three v4 files (arm), and hard2_train.
3. Base: `dagger.py --name novnoise-hard --no-velocity --rounds 3 --lidar-noise 0.2 --lidar-dropout 0.2 --balance --hard-scenarios scenarios_obstacles_hard2_train.json --select-scenarios scenarios_obstacles_hard2_select.json --data runs/obst-v2b runs/obst-v3-extra-0 ... runs/obst-hard2-train-*` (`train_base_torch.py` trains each round). Round 3 chosen on the hard2_select set, then checked against the previous model on fresh normal and hard layouts.
4. Arm: `train_arm.py` on the three v4 datasets, then `dagger_arm.py --name a --rounds 5` (expert parking; round 5 chosen), then `dagger_arm.py --name b --rounds 3 --base <final base> --init-model runs/dagger-arm-a/r5/model.pt --init-dagger runs/dagger-arm-a/r{1..5}-data` (round 3 chosen). Round-choice rules were fixed before each run.
5. Final checks: the evaluation commands above, on scenario sets not used to choose anything.

## Assignment scope

The source brief is **COMP0245-Assignment-2026-2.pdf (2026/27)**, read locally. See [assignment checklist](docs/assignment.md). Architecture, dataset size, splits, success thresholds and robustness scenarios here are project choices, not imposed assignment constraints.

## Setup

Use the existing course simulation environment. On the original workstation, `settings.json` points to the existing RoboEnv and course checkouts. On another machine:

1. Obtain [RoboEnv](https://github.com/VModugno/RoboEnv) with its submodules and [course materials](https://github.com/VModugno/LAB_SESSION_COMP_0245_2026_PUBLIC).
2. Follow RoboEnv's environment setup instructions. Keep its simulator assets external to this repository.
3. Install the additional packages in `requirements.txt` into that environment. Install PyTorch separately if using the experimental neural policies, choosing a build suitable for your machine.
4. Update `roboenv_root` and `course_root` in `settings.json`. Set `ROBOENV_PYTHON` to the environment's Python executable, or activate the environment and use `python pipeline.py` directly.

```bash
export ROBOENV_PYTHON=/path/to/RoboEnv/.pixi/envs/default/bin/python
bash run.sh doctor
bash run.sh collect --out runs/expert-v1
bash run.sh train --data runs/expert-v1 --model ridge --out runs/ridge-v1
bash run.sh evaluate --model-file runs/ridge-v1/model.joblib --split validation --out runs/ridge-v1-validation
```

Use fresh output directory names. `mlp` and `forest` are alternative baseline models. Add `--gui` to collection/evaluation for a simulator window. GUI operation and fresh-machine installation have not been validated for this snapshot. This simulation does not require ROS.

## Structure and outputs

- `pipeline.py`, `robot_env.py`, `models.py`: baseline collection, model fitting and closed-loop evaluation.
- `obstacle_env.py`, `make_obstacle_scenarios.py`, `collect_obstacles.py`: obstacle task, lidar, obstacle expert and data collection.
- `train_base_torch.py`, `dagger.py`, `eval_obstacles.py`, `robustness.py`, `rescore.py`: base policy training, DAgger and evaluation.
- `arm_data.py`, `train_arm.py`, `dagger_arm.py`, `eval_arm.py`: arm policy training, DAgger and evaluation.
- `demo.py`, `view_runs.py`: live PyBullet demonstration and browser replays. These are not the baseline CLI; see `--help`.
- `weights/`: the two final models.
- `scenarios*.json`, `settings.json`: scenario splits and experiment configuration.
- `tests/`: existing unit checks.
- `data/README.md`: collection and storage instructions.
- `docs/`: assignment requirements, evaluation, reproducibility and source attribution.
- `report/`: report and contribution-statement outlines.
- `runs/` (ignored): generated trajectories, manifests, models and evaluation outputs.

Baseline collection writes episode trajectories (`trajectory.npz`), episode metadata and a dataset manifest. Training writes `model.joblib` and `training.json`. Evaluation writes `evaluation.json`, `episodes.csv` and `error.png`. These are generated artifacts, not bundled evidence of performance.

## Validation

```bash
"$ROBOENV_PYTHON" -B -m unittest discover -s tests -v
```

Offline command prediction accuracy does not establish closed-loop success. See [evaluation and reproducibility](docs/evaluation.md). The legacy README is historical and contains descriptions of earlier model versions; current code and saved run configuration take precedence.

## External sources and authorship

The course expert is imported from `VModugno/LAB_SESSION_COMP_0245_2026_PUBLIC`, particularly `final/robotnik_arm_task/robotnik_arm_control.py`. Simulation is supplied by `VModugno/RoboEnv` and its `simulation_and_control` submodule. Those repositories are not vendored here. Project code adds collection, representations, regressors, learned rollout evaluation, obstacle experiments and reporting, developed with AI assistance. Review and understand it, and accurately record each student's contribution before submission. See [source register](docs/sources.md).

Start by tracing one observation/action pair through `robot_env.py`, then fit the ridge baseline and compare offline error with an actual rollout.
