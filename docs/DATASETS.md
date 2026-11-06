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
| `scenarios_obstacles_val_e.json` | 40 validation | `validatione_` | 30 Oct, seed 505: whole-robot robustness tests (comp0245/evaluation/robust_full.py) | 0.35-0.8 m | 1-3 | standard |
| `scenarios_obstacles_hard3.json` | 30 validation | `validationhard3_` | 30 Oct: expert-solved hard layouts (first 30 of 37 solved out of 48 candidates in `scenarios_obstacles_hard3_candidates.json`, seed 606); whole-robot robustness | 0.35-0.8 m | 2-5 | standard |
| `scenarios_obstacles_hard.json` | 30 validation | `validationhard_` | robustness: harder layouts never trained on | 0.35-0.8 m | 1-7 (mean 4), r 0.2-0.4 m, 3.5-5 m away | standard |
| `scenarios_obstacles_v4_height.json` | 200 train | `trainh_` | arm data: wider target heights only | **0.2-1.1 m** | 1-3 | standard |
| `scenarios_obstacles_v4_armstart.json` | 200 train | `traina_` | arm data: random arm start pose only | 0.35-0.8 m | 1-3 | **+-0.3 rad per joint** |
| `scenarios_obstacles_v4_both.json` | 200 train | `trainha_` | arm data: both changes | **0.2-1.1 m** | 1-3 | **+-0.3 rad** |
| `scenarios_obstacles_hard2_train.json` | 108 train | `trainhard2_` | hard layouts on purpose (`comp0245/scenarios/make_obstacle_scenarios.py --hard`): walls, 2-3 blockers, gaps down to 1 m, crowded parking spot; expert-solved only | 0.35-0.8 m | 2-6 | standard |
| `scenarios_obstacles_hard2_select.json` | 20 validation | `validationhard2s_` | same generator; used only to choose the DAgger round | 0.35-0.8 m | 2-5 | standard |
| `scenarios_obstacles_hard2_gate.json` | 30 validation | `validationhard2g_` | same generator; used only for the swap test | 0.35-0.8 m | 2-5 | standard |
| `scenarios_obstacles_moving_only.json` | 10 train | `trainmv_` | Julia: development set for the moving-obstacle expert; movers only | 0.35-0.8 m | movers, speed 0.1-0.3 m/s | standard |
| `scenarios_obstacles_moving_mixed.json` | 10 train | `trainmx_` | Julia: development set; movers plus static cylinders | 0.35-0.8 m | mixed | standard |
| `scenarios_obstacles_moving_test.json` | 1 train | `moving_cross_000` | Julia: single crossing-mover check (despite the file name, not a test split) | see file | 1 mover | standard |
| `scenarios_obstacles_moving_val_a.json` | 40 validation | `validationmva_` | 2 Nov, seed 1111: expert comparison (comp0245/evaluation/compare_experts.py); movers only, room half-size 5 | 0.35-0.8 m | 2-3 movers | standard |
| `scenarios_obstacles_moving_val_b.json` | 40 validation | `validationmvb_` | 2 Nov, seed 1112: expert comparison; mixed, room half-size 5 | 0.35-0.8 m | 1-2 movers + 1-3 static | standard |
| `scenarios_rooms_train.json` | 200 train | `trainrmv_`, `trainrmx_` | 2 Nov, seeds 2101, 2102 (runs/rooms-collect.sh): moving-base training; rooms 7-14 m wide, layout at a random place in its room; `rmv` 2-3 movers, `rmx` 1-2 movers + 1-3 static | as v2 | see purpose | fixed |
| `scenarios_rooms_train2.json` | 600 train | `trainrmv2_`, `trainrmx2_` | 5 Nov, seeds 2111, 2112: 600 more training rooms, same settings as `scenarios_rooms_train.json` (overnight run). Crossing variants: `runs/crossing/scenarios_crossing_train2.json` (599) | as v2 | see purpose | fixed |
| `scenarios_rooms_val.json` | 60 validation | `validationrmv_`, `validationrmx_` | 2 Nov, seeds 2103, 2104: same recipe | as v2 | see purpose | fixed |
| `scenarios_rooms_test.json` | 80 test | `testrmv_`, `testrmx_` | 2 Nov, seeds 2105, 2106: same recipe. **Reserved for the final evaluation, not run** | as v2 | see purpose | fixed |
| `scenarios_floorplan_val.json` | 40 validation | `fp_` | 2 Nov, seed 707: expert comparison; random floor plans, 8 x 8 m, 28 with a mover | 0.35-0.8 m | furniture, up to 1 mover | standard |

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
| `arm-a` | arm model A (comp0245/training/train_arm.py, 24 inputs, gravity + network), trained by Akhilesh on the three obst-v4 sets |
| `arm-model-b`, `arm-eval-model-b` | 3 Nov: model B (`comp0245/training/train_arm.py --ik`, 31 inputs) on the three obst-v4 sets, and its closed-loop test on the same 90: 42/90 (height 9/20, armstart 6/20, both 7/20, fresh 20/30) against arm-a's 62/90 |
| `arm-eval-model-a-check` | 3 Nov: weights/arm_final.pt on 12 of the 90 after the model B wiring: all 12 identical to `dagger-arm-b/r3-eval` |
| `arm-eval-arm-a`, `arm-eval-expert` | closed-loop arm test (comp0245/evaluation/eval_arm.py): expert wheels + learned arm, and the full-task expert on the same scenarios |
| `dagger-arm-a` | arm DAgger with a shadow expert (comp0245/training/dagger_arm.py), rounds r0-r5; `chosen.json` = r5 (84/90); starting point for dagger-arm-b |
| `arm-eval-full`, `arm-eval-full-hybrid`, `arm-eval-full-expert` | final check on validationc_060-099: fully learned robot 23/40, expert wheels + arm r5 40/40, full-task expert 40/40 |
| `dagger-arm-b` | arm DAgger with the LEARNED base parking (final base), from dagger-arm-a r5 + its data; r0-r3; fully learned / expert parking on the 90: 41/84, 47/76, 66/86, 77/87; `chosen.json` = r3; **`r3/model.pt` = final arm** |
| `arm-eval-final`, `arm-eval-final-hybrid`, `arm-eval-final-expert` | final check on the new `scenarios_obstacles_val_d.json` (validationd_000-039): fully learned 37/40 (contact 0), expert wheels + arm 40/40, full-task expert 40/40 |
| `arm-eval-rooms-full-val` | 5 Nov: full robot in the 60 validation rooms (room model r3-hold-latch + arm_final, arm folded, unfold before reaching, 100 s): 32/60 |
| `final-test/base`, `arm-eval-final-test` | 5 Nov: THE single final test evaluation on `scenarios_rooms_test.json` (80 rooms): base only 48/80 parked (70 s), full robot 45/80 (100 s). Evaluation only, never training data |
| `arm-parked/` | 5 Nov: arm retrained for the room robot. `scenarios.json` 1,000 parked spots (height 0.36-0.80 m, bearing +-15 deg, 80-100% of the reach limit); `data` the expert arm's reaches from them (1000/1000); `arm` the new arm (old arm data + DAgger + this); `test-current`, `test-new` the arm-only test on the 49 validation parking spots (25 vs 38). Full robot with it: `arm-eval-rooms-full-val-newarm` 37/60 |
| `arm-parked/dagger-r1`, `-r2`, `synth200.json`, `test-new-synth200` | 5 Nov: arm DAgger on parked spots (500 spots per round, shadow-expert labels, `runs/arm-parked/dagger.py`). r2 adopted: 45/49 validation spots, 188/200 on `synth200.json` (200 held-out synthetic spots, seed 777; room arm before DAgger 103/200). Full robot: `arm-eval-rooms-full-val-dagger2` 37/60 |
| `obst-hard3-a/-b/-c` | expert drives on the 48 hard3 candidates (37 solved) |
| `robust-full` | 30 Oct: whole robot (final base + final arm) vs full-task expert: clean 37/40 vs 40, lidar 38 vs 40, push 35 vs 35, delay 38 vs 40, hard3 22/30 vs 30 (comp0245/evaluation/robust_full.py) |
| `visible-expert` | 2 Nov: full RRT expert 119/120 vs visible-only 117/120 on the three sets above (comp0245/evaluation/compare_experts.py); no contact with walls |
| `rooms-expert` | 2 Nov: visible-only RRT expert on scenarios_rooms_train.json, 110 s limit, 183/200 clean, 6.99 M rows at 1 kHz, 4.0 GB. Failures stay in the folder; training uses successful episodes only. `obstacle_seen` = a lidar ray hit that obstacle at that step |
| `rooms-expert-val` | 2 Nov: same expert on scenarios_rooms_val.json, 54/60 clean, 1.2 GB. The reference to compare a moving model against |
| `rooms-trial` | 2 Nov: 16-episode trial before the placement fix; not training data |
| `robust-recheck-2nov` | 2 Nov: final model on val_e with the current code: 37/40, all 40 episodes identical row for row to `robust-full` |
| `scans-rooms` | 2-3 Nov: comp0245/evaluation/compare_scans.py, plain cloning on `rooms-expert`, 1 vs 3 scans, 2 seeds, driven on the 60 room validation scenarios (results.json). `base-final-rerun`: `weights/base_final.pt` on the same 60. `base-final` is empty, left by a run that crashed |
| `scans-rooms-baseline.log.gz` | 3 Nov: only the repeated worker crash from that run, kept compressed until Akhilesh decides. No results in it |
| `dagger-rooms-trial` | 3 Nov: 10 shadow-labelled room runs (`label`) and 3 base_final runs that matched the baseline row for row (`same`); a check, not training data |
| `dagger-rooms-1scan`, `dagger-rooms-3scans`, `dagger-rooms-velocity` | 3 Nov (runs/dagger-rooms.sh): room DAgger with the visible-only RRT as shadow expert, 3 rounds x 100 runs on scenarios_rooms_train.json, scored on the 60 room validation scenarios. r0 of the first two is copied from `scans-rooms/scans{1,3}-seed0`. Labelling runs stop after 2 s of contact or 20 s still; all kept. `results.json` per folder |
| `julia-dataset/dataset` | Julia's 1,650 expert runs (from `/data/comp0245/dataset.tar`, extracted 3 Nov, tar kept). **Full-knowledge RRT** (`"expert": "rrt"`), saved every 10th step. case1: 150, empty 10 x 10 m room. case2: 400, 1-4 cylinders, 178 with movers. case3: 300, 8 x 8 m floor plans, no movers. case4: 500, floor plans + obstacles, 332 with movers. case5: 300, floor plans, several dots, 146 with movers. 1,625 successful. Scenarios in `caseN.json` with her own train / validation / test splits: use her train split only. No `manifest.json` and no obstacle positions saved, so our loader needs a manifest and threat labels would need the movers replayed |
| `dagger-rooms-conv-threat` | 3 Nov (runs/dagger-rooms-2.sh): room DAgger as `dagger-rooms-3scans` but with the learned ray filter and the threat output; round 0 trained here. r3 12 strict / 16 loose / 28 no contact |
| `dagger-rooms-conv`, `dagger-rooms-threat` | 3 Nov (runs/dagger-rooms-4.sh): the same with the ray filter only / threat output only |
| `reach-area`, `reach-area-fine` | 3 Nov (comp0245/evaluation/measure_reach.py): expert arm from a still base, no obstacles; target at 0.3-1.2 m (0.1 m steps) and 0.90-1.10 m (0.02 m steps), straight ahead and 30 deg to the side, 0.36 / 0.58 / 0.80 m high. Defines the reach area of the "parked" score |
| `rescored.json` in room evaluation folders, `rooms-expert-val`, `robust-hard-r3`; `rescore-parked-3nov.log` | 3 Nov: the same runs scored with "parked" (comp0245/evaluation/rescore.py); evaluation.json / results.json untouched. A folder that already had a rescored.json from 28 Oct would get rescored-parked.json instead |
| `dagger-rooms-mix`, `rooms-expert-steady(-val)`, `dagger-rooms-steady` | 3 Nov (runs/dagger-rooms-5.sh): room DAgger with the ray filter plus a mixture output; steadier-expert data on the same room scenarios (110 s, `--rrt --steady`), then room DAgger labelled by that expert |
| `dagger-arm-modelb`, `dagger-arm-modelb-lb`, `arm-eval-final-modelb(-hybrid)` | 3 Nov: arm DAgger for model B with expert parking (r0-r5, 42 -> 88/90), then with the learned base parking (round chosen by runs/choose_arm_round.py), then the val_d check like the final arm's |
| `modes-gate` | 3 Nov (side session lidar72-modes): offline gate for the evade/target switch. `modes/` (--modes --mode-label collision) and `plain-conv/` (same settings without modes): conv, 3 scans, no velocity, 36 rays, 10 epochs on `rooms-expert`, validation `rooms-expert-val`. `gate.json` (gate_check.py), `switch_diag.json` (switch_diag.py: visibility of the threat, switch trained alone), `closed-loop-smoke/` (modes model on 3 room validation scenarios, with modes.npz per run). Not a model to keep |
| `crossing/`, `dagger-rooms-crossing`, `dagger-rooms-crossing-s1` | 4 Nov: crossing DAgger from window r3 with the arm folded (runs/crossing/*.sh): each round 50 normal + 50 crossing-mover runs (`crossing/scenarios_crossing_train.json`), window-expert labels, 3 rounds; seed 0 and seed 1. `gate.json` = copy score before window r3's real hits plus parked / mover hits per round. Seed 0 r3 35 / 12, seed 1 r3 32 / 13 |
| `dagger-rooms-crossing-r4`, `crossing-r4-start` | 4 Nov: a 4th crossing round from seed 0 r3 (seed 3). 35 parked / 14 mover hits; not kept |
| `rooms-window72-new` | 5 Nov overnight: window expert on the 600 rooms of `scenarios_rooms_train2.json`, arm folded, 72 rays (373/600 successful) |
| `dagger-rooms-big-s0`, `dagger-rooms-big-s1`, `overnight-rooms` | 5 Nov overnight: crossing DAgger on all 800 training rooms with the mirror flip and early stopping, 2 seeds; only r3 validated. 28 / 15 and 36 / 15 parked / mover hits; not kept. `overnight-rooms` has run.sh, status.sh and the combined crossing pool |
| `crossing-hold/` | 4 Nov: learned hold output (comp0245/training/train_hold.py) on frozen crossing r3 models. `r3-hold` (offline check `hold_check.json`), `r3-hold-latch` = **adopted room model** (37 / 12), `r3-hold-latch-validation-100s` (report-only 100 s check, 39), `s1-r3-hold(-latch)` = the same for seed 1 (31 / 13) |
| `efficiency.json` | parameters, file size, per-step CPU time of the final models (comp0245/evaluation/efficiency.py) |
| `robust-base-r2`, `robust-noise-r2` | overnight: robustness tests (results.json has the tables) |
| `robust-fix-expert` | 29 Oct: the like-for-like expert reference (`expert_hold`: expert wheels, same fixed arm hold and 40 s as the learned models) on every test |
| `robust-fix-delay` | 29 Oct: delay tests rerun for the four round-2 models after the delay warm-up fix (supersedes the delay rows of the two overnight runs) |

Scoring: evaluations scored before 29 Oct have their original `results.json` / `evaluation.json` plus a
`rescored.json` from `comp0245/evaluation/rescore.py` (parking must hold for the last 2 s, base stopped, including yaw rate). Use
`rescored.json`. Since 4 Nov room evaluations also have `rescored-reach.json` (measured-reach parking rule, the
reported "parked"; `parked_worst_case_reach` keeps the older 83% rule). The `expert` rows in `robust-base-r2` are the full-task expert (arm reaches, episode ends early):
not a like-for-like reference; use `robust-fix-expert`.

Inside a DAgger folder: `rK/model.pt` (model after round K), `rK-validation/` (closed-loop scores on the 10
reference validation scenarios), `rK-data/` (drives by the round K-1 model, labelled by the expert; saved at 100
rows per second), `results.json` (one row per round).
