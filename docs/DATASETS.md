# Datasets and scenario files

What every scenario file and recorded dataset is, so settings can be mixed and matched. Scenario files are in the
project folder; recorded runs are in `runs/` (stored on /data). Superseded ones are in `archive/` (see
archive/ARCHIVE_NOTE.md). All obstacle data uses the obstacle expert (obstacle_env.SteerAround), wheel sideways
grip 0.1 (course: 0.01) and reach-zone parking, unless stated.

## Scenario files (inputs to collection and evaluation)

| File | Count | Ids | Purpose | Targets | Obstacles | Arm start |
|---|---|---|---|---|---|---|
| `scenarios_random_v1.json` | 60/10/10 | `train_`, `validation_`, `test_` | course task, no obstacles (dataset-v3) | height 0.35-0.8 m | none | standard |
| `scenarios_obstacles_v2.json` | 60/10/10 | `train_`, `validation_`, `test_` | main obstacle set; its 10 validation scenarios are the reference for all DAgger comparisons; **test = reserved for the final evaluation** | 0.35-0.8 m | 1-3 | standard |
| `scenarios_obstacles_v3_extra.json` | 160 train | `trainb_` | more training variety, same settings as v2 | 0.35-0.8 m | 1-3 | standard |
| `scenarios_obstacles_val100.json` | 100 validation | `validationc_` | fresh validation pool, now fully used: 0-29 robustness tests, 30-59 arm round choice (eval_arm fresh-normal), 60-99 first fully-learned check | 0.35-0.8 m | 1-3 | standard |
| `scenarios_obstacles_val_d.json` | 40 validation | `validationd_` | 29 Oct, seed 404: final fully-learned check of dagger-arm-b r3 (used once) | 0.35-0.8 m | 1-3 | standard |
| `scenarios_obstacles_hard.json` | 30 validation | `validationhard_` | robustness: harder layouts never trained on | 0.35-0.8 m | 1-7 (mean 4), r 0.2-0.4 m, 3.5-5 m away | standard |
| `scenarios_obstacles_v4_height.json` | 200 train | `trainh_` | arm data: wider target heights only | **0.2-1.1 m** | 1-3 | standard |
| `scenarios_obstacles_v4_armstart.json` | 200 train | `traina_` | arm data: random arm start pose only | 0.35-0.8 m | 1-3 | **+-0.3 rad per joint** |
| `scenarios_obstacles_v4_both.json` | 200 train | `trainha_` | arm data: both changes | **0.2-1.1 m** | 1-3 | **+-0.3 rad** |
| `scenarios_obstacles_hard2_train.json` | 108 train | `trainhard2_` | hard layouts on purpose (`make_obstacle_scenarios.py --hard`): walls, 2-3 blockers, gaps down to 1 m, crowded parking spot; expert-solved only | 0.35-0.8 m | 2-6 | standard |
| `scenarios_obstacles_hard2_select.json` | 20 validation | `validationhard2s_` | same generator; used only to choose the DAgger round | 0.35-0.8 m | 2-5 | standard |
| `scenarios_obstacles_hard2_gate.json` | 30 validation | `validationhard2g_` | same generator; used only for the swap test | 0.35-0.8 m | 2-5 | standard |

Standard arm start pose = the course's [0, 1.57, 0, 1, 0, 1, 0] rad.

## Recorded expert datasets (`runs/`)

| Folder | From scenario file | Episodes | Notes |
|---|---|---|---|
| `dataset-v3` | scenarios_random_v1 | 80 | course expert, no obstacles, course physics (grip 0.01) |
| `obst-v2b` | scenarios_obstacles_v2 | 80 | current obstacle expert; validation + test inside |
| `obst-v3-extra-0..3` | scenarios_obstacles_v3_extra | 4 x 40 | same expert, extra training variety |
| `obst-v4-height`, `obst-v4-armstart`, `obst-v4-both` | the three v4 files | 200 each | overnight 28-29 Oct; arm data |

## Learned-model runs (`runs/`)

| Folder | What |
|---|---|
| `archive/runs/base-mlp-v1(-validation)` | first sklearn base model (no obstacles); archived 29 Oct |
| `archive/runs/obst-base-torch-v1(-validation)`, `archive/runs/obst-base-torch-v1-novel(-validation)` | first PyTorch lidar models (60 episodes), with / without velocity; archived 29 Oct |
| `dagger-nov`, `dagger-vel2` | DAgger without / with velocity, 220 expert episodes, rounds r0-r3 (r0 = plain BC) |
| `archive/runs/dagger-vel` | interrupted first attempt (round 0 only); archived 29 Oct |
| `dagger-velnoise`, `dagger-novnoise` | overnight: noise-robust DAgger (`dagger-novnoise/r2` was the final model 29 Oct morning) |
| `dagger-novnoise-hard` | noise DAgger with hard layouts mixed in (half the drives) and a balanced loss; **`r3/model.pt` = final base model** (passed the swap test 29 Oct) |
| `obst-hard2-train-a/-b`, `obst-hard2-select`, `obst-hard2-gate` | expert drives on the hard-layout candidates (only solved ones are used) |
| `robust-gate-normal`, `robust-gate-oldhard`, `robust-gate-newhard` | swap test: novnoise r2 vs novnoise-hard r3 |
| `robust-hard-r3` | full robustness suite on the final base model (29 Oct afternoon) |
| `arm-a` | arm model A (train_arm.py, 24 inputs, gravity + network), trained by Akhilesh on the three obst-v4 sets |
| `arm-eval-arm-a`, `arm-eval-expert` | closed-loop arm test (eval_arm.py): expert wheels + learned arm, and the full-task expert on the same scenarios |
| `dagger-arm-a` | arm DAgger with a shadow expert (dagger_arm.py), rounds r0-r5; `chosen.json` = r5 (84/90); starting point for dagger-arm-b |
| `arm-eval-full`, `arm-eval-full-hybrid`, `arm-eval-full-expert` | final check on validationc_060-099: fully learned robot 23/40, expert wheels + arm r5 40/40, full-task expert 40/40 |
| `dagger-arm-b` | arm DAgger with the LEARNED base parking (final base), from dagger-arm-a r5 + its data; r0-r3; fully learned / expert parking on the 90: 41/84, 47/76, 66/86, 77/87; `chosen.json` = r3; **`r3/model.pt` = final arm** |
| `arm-eval-final`, `arm-eval-final-hybrid`, `arm-eval-final-expert` | final check on the new `scenarios_obstacles_val_d.json` (validationd_000-039): fully learned 37/40 (contact 0), expert wheels + arm 40/40, full-task expert 40/40 |
| `robust-base-r2`, `robust-noise-r2` | overnight: robustness tests (results.json has the tables) |
| `robust-fix-expert` | 29 Oct: the like-for-like expert reference (`expert_hold`: expert wheels, same fixed arm hold and 40 s as the learned models) on every test |
| `robust-fix-delay` | 29 Oct: delay tests rerun for the four round-2 models after the delay warm-up fix (supersedes the delay rows of the two overnight runs) |

Scoring: evaluations scored before 29 Oct have their original `results.json` / `evaluation.json` plus a
`rescored.json` from `rescore.py` (parking must hold for the last 2 s, base stopped, including yaw rate). Use
`rescored.json`. The `expert` rows in `robust-base-r2` are the full-task expert (arm reaches, episode ends early):
not a like-for-like reference; use `robust-fix-expert`.

Inside a DAgger folder: `rK/model.pt` (model after round K), `rK-validation/` (closed-loop scores on the 10
reference validation scenarios), `rK-data/` (drives by the round K-1 model, labelled by the expert; saved at 100
rows per second), `results.json` (one row per round).
