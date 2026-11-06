# UCL COMP0245 — Machine Learning for Robotics

Behaviour cloning for a mobile manipulator: learn commands from expert trajectories to reach Cartesian end-effector targets and evaluate generalisation to new targets and initial conditions.

This repository starts from Akhilesh's existing project code, including baseline regression and experimental PyTorch base/arm policies. It is a development snapshot, not a claim of completed assessment or successful final policies. No datasets, weights or measured results are bundled.

## Assignment scope

The source brief is **COMP0245-Assignment-2026-2.pdf (2026/27)**, read locally. See [assignment checklist](docs/assignment.md). Architecture, dataset size, splits, success thresholds and robustness scenarios here are project choices, not imposed assignment constraints.

## Project layout

```text
comp0245/               the Python package (run modules with `python -m comp0245.<sub>.<module>`)
  paths.py              fixed repository locations (ROOT, config, scenarios, weights, runs)
  models.py, pipeline.py, arm_data.py
  preprocess.py         lidar features: sectors, walls, openings and doorways, blocked line to each dot
  sim/                  simulator wrappers: robot, obstacles, lidar, RRT and A* experts
  scenarios/            scenario generators (make_dataset.py: the five cases) and previews
  training/             expert data collection, MLP training, DAgger, feature caching
  evaluation/           closed-loop evaluation (eval_mlp.py), arm tests, robustness
  apps/                 drive_mlp.py, interactive_test.py, view_runs.py (HTML template in apps/templates/)
config/settings.json    simulator paths, control and success settings
data/scenarios/         scenario files (*.json); a bare file name on the command line is found here
weights/                saved model (wheels_mlp_dagger_pre_r3.pt) and its result
tests/                  unit tests, plus check_*.py scripts that compare features with the true layouts
docs/, report/          notes and the report draft
runs/                   outputs (not in Git)
```

Run every command from the repository root, for example `python -m comp0245.apps.drive_mlp --case case3 --gui`.
Tests: `python -B -m unittest discover -s tests -t .`. Code style: PEP 8, checked with `ruff format .` and
`ruff check .` (settings in `pyproject.toml`; `uvx ruff ...` runs it without installing).

## Setup

Use the existing course simulation environment. On the original workstation, `config/settings.json` points to the existing RoboEnv and course checkouts. On another machine:

1. Obtain [RoboEnv](https://github.com/VModugno/RoboEnv) with its submodules and [course materials](https://github.com/VModugno/LAB_SESSION_COMP_0245_2026_PUBLIC).
2. Follow RoboEnv's environment setup instructions. Keep its simulator assets external to this repository.
3. Install the additional packages in `requirements.txt` into that environment. Install PyTorch separately if using the experimental neural policies, choosing a build suitable for your machine.
4. Update `roboenv_root` and `course_root` in `config/settings.json`. Set `ROBOENV_PYTHON` to the environment's Python executable, or activate the environment and use `python -m comp0245.pipeline` directly.

```bash
export ROBOENV_PYTHON=/path/to/RoboEnv/.pixi/envs/default/bin/python
bash run.sh doctor
bash run.sh collect --out runs/expert-v1
bash run.sh train --data runs/expert-v1 --model ridge --out runs/ridge-v1
bash run.sh evaluate --model-file runs/ridge-v1/model.joblib --split validation --out runs/ridge-v1-validation
```

Use fresh output directory names. `mlp` and `forest` are alternative baseline models. Add `--gui` to collection/evaluation for a simulator window. GUI operation and fresh-machine installation have not been validated for this snapshot. This simulation does not require ROS.

## Structure and outputs

- `comp0245/pipeline.py`, `comp0245/sim/robot_env.py`, `comp0245/models.py`: baseline collection, model fitting and closed-loop evaluation.
- `comp0245/training/train_base_torch.py`, `comp0245/training/train_arm.py`, `eval_*.py`, `dagger*.py`, `comp0245/sim/obstacle_env.py`: experimental base/arm and obstacle extensions. Inspect `--help` before use; these are not the baseline CLI.
- `scenarios*.json`, `config/settings.json`: scenario splits and experiment configuration.
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

Start by tracing one observation/action pair through `comp0245/sim/robot_env.py`, then fit the ridge baseline and compare offline error with an actual rollout.
