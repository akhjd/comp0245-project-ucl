# Robot learning pipeline

**Working location on this PC:** `/home/akhjd/ros2_ws/comp0245_project`. Open a terminal here for ongoing work. The earlier Documents/Codex copy is a bootstrap snapshot.

A small, interchangeable behaviour-cloning pipeline for the COMP0245 Robotnik mobile manipulator. It uses your existing RoboEnv Pixi environment and the course expert controller. No packages are installed and no external repositories are edited.

The existing cloned dependencies are siblings: `../RoboEnv` holds `pixi.toml`, `pixi.lock`, the installed environment and simulator assets; `../LAB_SESSION_COMP_0245_2026_PUBLIC` holds the supplied expert. This project owns the training/evaluation code and runs. It currently reuses RoboEnv’s environment; an independent environment for sharing is a separate portability step.

## Start here

Open a terminal in this folder. Run commands with `bash run.sh`. The launcher uses the installed RoboEnv Python and keeps plots/caches inside `runs/`.

An already collected pilot dataset is in `runs/dataset-v2`. Trained initial models and their results are listed in `RUN_STATUS.md`. You can train another model on this dataset immediately; choose a fresh output name:

```bash
bash run.sh train --data runs/dataset-v2 --model mlp --out runs/my-mlp-v2
bash run.sh evaluate --model-file runs/my-mlp-v2/model.joblib --split validation --out runs/my-mlp-v2-validation
```

For a new collection and model from scratch:

```bash
bash run.sh doctor
bash run.sh collect --out runs/my-dataset-v3
bash run.sh train --data runs/my-dataset-v3 --model mlp --out runs/my-mlp-v3
bash run.sh evaluate --model-file runs/my-mlp-v3/model.joblib --split validation --out runs/my-mlp-v3-validation
```

Switch the model without changing the simulator or dataset:

```bash
bash run.sh train --data runs/dataset-v2 --model ridge --out runs/my-ridge-v2
bash run.sh evaluate --model-file runs/my-ridge-v2/model.joblib --split validation --out runs/my-ridge-v2-validation
bash run.sh train --data runs/dataset-v2 --model forest --out runs/my-forest-v2
```

Use validation episodes for development. After choosing the model/hyperparameters, evaluate the reserved test split once:

```bash
bash run.sh evaluate --model-file runs/mlp-v1/model.joblib --split test --out runs/mlp-v1-test
```

Add `--gui` to collect/evaluate to watch the simulated robot on a desktop with a working display. Headless runs are the default. GUI operation needs a separate check on your desktop. Headless simulation does not require Sunshine/Moonlight or ROS.

Existing run directories are never overwritten. Use a new versioned name for each experiment. Ctrl+C stops a run; a partial dataset manifest includes only completed episodes. Failed runs retain diagnostics in `failure.json`.

## Where to change things

| File | Purpose |
|---|---|
| `settings.json` | External repository paths, control frequency, history length, success threshold, model hyperparameters |
| `scenarios.json` | Fixed target coordinates, IDs, seeds, and whole-episode split assignments |
| `comp0245/models.py` | Observation/action names, causal history, preprocessing and model registry |
| `comp0245/sim/robot_env.py` | Simulator adapter, expert integration and episode metrics |
| `comp0245/pipeline.py` | Collection, training, evaluation and run manifests |

To use a different machine, update repository paths in `settings.json` and set `ROBOENV_PYTHON` to that machine's working environment interpreter. The launcher defaults to the existing Linux installation; Windows/macOS launchers have not been tested. Python execution also works directly from a properly activated RoboEnv environment.

## The shared model interface

An estimator implements `fit(X, Y)` and `predict(X)`. Register a new estimator in `make_model()` and expose its name in the CLI choices. Every model receives the same features, episode split and output commands.

One observation has **28 numerical features**: base world position, sine/cosine of base yaw, target world position, seven arm joint positions, seven arm joint velocities, base linear velocity and base angular velocity (world frame). No expert controller phase or internal arm reference is passed to the learned policy. Three causal frames are concatenated by default (84 inputs), with the first frame repeated at episode start. Each frame is 1 ms apart. This short history is an initial engineering choice, not evidence of adequate memory or an optimal representation. Target-relative inputs and longer histories are future comparisons.

The 11 outputs are four wheel angular velocities (rad/s, front-right/front-left/back-left/back-right) and seven arm torques (N m). Gripper fingers remain fixed using the provided course helper. Predictions are clipped to the configured limits, with clipping frequency reported. The first model attempts the full output interface; there is no hidden expert fallback during learned rollouts.

`ridge`: regularised linear reference. `mlp`: small two-layer neural network. `forest`: multi-output random forest. These are initial comparison candidates, not a literature-backed claim that one is best.

Inputs and outputs are standardised using training episodes only. Each episode's history is built separately; it cannot leak across episode boundaries. Test observations/actions are not loaded during training or offline validation. All models use the same units and observation/action order.

The model artifact stores preprocessing, estimator, feature schema, training scenario IDs, collection settings and dataset-manifest hash. Only load your own trusted `model.joblib` files: joblib/pickle loading can execute code.

## Data your teammate can merge

Each episode contains:

- `trajectory.npz`: pre-action observations, corresponding raw/applied actions, simulation timestamps, end-effector/base positions, arm positions, errors and inference timings.
- `episode.json`: scenario, collection settings, hashes of course/config sources and terminal end-effector position.
- `summary.json`: episode outcome and metrics.
- `simulator.log`: simulator/controller diagnostics.

Every training row pairs the observation measured immediately before an action with that action. Physics and expert/learned commands run at 1 kHz, matching the original task's update rate. An initial 200 Hz experiment made the expert unstable, so 1 kHz was retained. Changing the action rate requires collecting a new compatible dataset and verifying expert stability again. The success hold criterion and explicit time budget are runner additions.

The dataset manifest lists feature/action names, schema version and episode split. The starter suite has four training targets, one validation target and one test target, all with the same configured starting pose. Different seeds alone do not guarantee different trajectories in this deterministic simulation. This is a smoke/pilot dataset, not enough to establish broad generalisation.

Only successful expert episodes are used for training and offline validation. Failed episodes stay on disk for inspection. If no successful training/validation episodes exist, training stops rather than silently fitting failed demonstrations. For teammate-produced data, agree on this schema, action rate and target splits before merging; importing another schema needs an explicit adapter. The CLI does not silently merge incompatible datasets.

## Evaluation and current limits

Success means reaching within 1 cm for at least 0.25 simulated seconds, after at least 8 simulated seconds, within the configured episode time budget. The 1 cm threshold comes from the course starter; dwell/minimum time and 35 s budget are explicit runner choices. A timeout is always a failure. There is no physical-unreachability diagnosis.

Evaluation saves per-episode CSV, JSON summary and an end-effector-error plot. Metrics include final/minimum position error, success, time, base path length, total arm joint travel, action clipping and inference median/p95. Failed rollouts remain in summary statistics. Success criteria are independent of the expert's navigation phase. Expert base navigation timeout is recorded separately.

Collision/obstacle metrics are **not implemented** and are recorded as null, never as zero collisions. There are no added sensors or obstacles yet. This is a baseline pipeline, not object detection or avoidance. There is no end-effector orientation target. There is no GPU requirement for these initial CPU estimators.

A good offline prediction error does not imply successful closed-loop control. Distribution shift, hidden expert memory, too little target diversity and direct torque prediction can cause failure. Report actual rollout results, including failures. Use validation for iterations and keep a fresh untouched test set if you repeatedly inspect the current test results.

## Checks and learning route

```bash
# From a RoboEnv terminal, or using its Python directly:
python -B -m unittest discover -s tests -v
```

Suggested walkthrough: inspect one observation/action row in `comp0245/models.py`; follow one simulator step in `comp0245/sim/robot_env.py`; inspect `training.json`; then compare its offline errors with the validation rollout. Change one hyperparameter at a time and keep separate output folders.

Your next significant work can be a target-relative feature encoder or a dataset-import adapter. Those are focused, understandable additions you can write yourself and merge with teammates. This folder can become your team's Git repository after agreeing its location and attribution. Nothing is pushed or published automatically.

## Source attribution

The expert implementation is imported from `VModugno/LAB_SESSION_COMP_0245_2026_PUBLIC`, `final/robotnik_arm_task/robotnik_arm_control.py`. Simulation comes from `VModugno/RoboEnv` and its `simulation_and_control` submodule. This project adds the adapter, observation history, dataset format, model registry, train/evaluate CLI and reporting. The existing course/config sources are hashed per episode. Keep this attribution in any submitted or shared README and record subsequent student changes accurately.
