# Project checklist

Deadline: report + code 6 Nov 2026. Lab demo: any session 30 Oct - 6 Nov (needed to complete the assessment).
Order matters: anything that changes the task (room, moving obstacles) needs new expert data, retraining and
re-evaluation, so it comes BEFORE the design is frozen. The test evaluation is the very last measurement.

Status: [x] done · [ ] to do · [?] to decide
Dates, order, stop rules and the reasoning for each item: ROADMAP.md (updated 2 Nov).
Julia's `rrt-expert` work (00984f1) is merged locally, not on GitHub main yet: expert side of the room, moving cylinders, floor plans, A* and several dots.

## 1. Current system (done)
- [x] Learned base: lidar + target, noise-robust DAgger with hard layouts (weights/base_final.pt)
- [x] Learned arm: gravity + network, DAgger with a shadow expert, then with the learned base parking (weights/arm_final.pt)
- [x] Fully learned robot on fresh validation: 37/40, no collisions (expert 40/40)
- [x] Live demo (comp0245/apps/demo.py) and browser replays (comp0245/apps/view_runs.py, runs/viewers.html)
- [x] GitHub repo updated 29 Oct (newer local work not pushed yet: see section 5)

## 2. Model B: IK direction as extra arm inputs (handed to Claude 3 Nov)
- [x] `ik_direction` in comp0245/arm_data.py + tests (21 of 21 pass)
- [x] Arm inputs 24 -> 31 (arm_data.arm_inputs_b, `comp0245/training/train_arm.py --ik`), trained on the same data (runs/arm-model-b)
- [x] Compare A vs B (runs/arm-eval-model-b): B 42/90 against A 62/90, so A stays. 32 of B's 48 misses got within 3 cm and then moved away
- [x] Arm DAgger for B anyway (Akhilesh: to be sure; runs/dagger-arm-modelb): 42, 64, 76, 76, 79, **88**/90 (A's run: 62 ... 84; final arm 87)
- [x] B's second stage (runs/dagger-arm-modelb-lb): 69/90 fully learned (final arm 77), val_d 37/40 fully learned, 38/40 expert base (final arm 37 / 40). B does not beat it: arm_final stays

## 3. New environment features
Decided 29 Oct (Akhilesh): model B -> bounded room -> moving cylinders, in that order. Multiple robots would run
our policy (with some changes), not scripted paths. SLAM is very hard but doable: last, if time allows.
Each needs: environment change, expert that still solves it, new data, retraining, new scenario sets.
- [ ] **Bounded room** (expert done on Julia's branch; learned 30/40 with no retraining): walls around the arena instead of infinite space. Lidar already sees walls;
      the expert must not hug or hit them; spawn positions and targets must fit inside.
- [ ] **Moving obstacles** (expert done on Julia's branch, privileged RRT; learned 0/10 moving, 2/10 mixed with no retraining). The current base sees ONE lidar scan, so it cannot
      tell a moving obstacle from a still one: needs stacked recent scans (temporal input, which the brief
      explicitly asks us to consider). The expert must also cope (it is reactive; may need prediction).
- [ ] **Very fast obstacles**: test only, the top of a success-vs-speed sweep (some cannot be dodged by anything).
- [ ] **Edge cases**: unsolvable targets, already in reach, start near walls or in corners, out-of-reach targets.
- [ ] **Apartment + A\***: expert done on Julia's branch; learned 1/20 with no retraining. A* stays in the expert only; no A* while the model drives (decided 2 Nov).
- [ ] **Several dots**: expert done on Julia's branch (order, arm return). Learned version needs the arm return outside the expert.
- [ ] **Red balls into a box**: demo built on several dots; attached by a constraint (fixed fingers), balls on stands.
- [ ] **Find initial position**: drift test on the learned robot; particle filter dropped (faculty prefer learned over built-in code).
- [?] **Multiple robots, different goals**: copies of our policy; each sees the others only through lidar. Needs stacked scans.
- [?] **Multiple robots, same goal**: only if time is left; needs a rule for taking turns.
- [?] **Fall detection**: stop and report on large tilt. Proposed instead of self-righting (about 150 N m needed; arm capped at 30).
- [x] **DAgger with the RRT expert** (3 Nov, runs/dagger-rooms-*): shadow visible-only expert, 3 rounds. Parking better (1 scan loose 3 -> 9 of 60), mover contact not (36-45 of 60 everywhere); still below base_final's 9 strict / 14 loose
- [x] **Visible-only expert** (`VisibleRRTExpert`): 117/120 vs 119/120 for the full expert; default for `--rrt` since 2 Nov.
- [x] **Moving-base data in mixed-size rooms:** `runs/rooms-expert` 183/200 clean, `runs/rooms-expert-val` 54/60; test file written, not run
- [ ] Expert weakness: repeated failed plans near a visible mover (18 of 23 failures); look at it before DAgger
- [x] **Threat prediction + learned ray filter** (3 Nov, runs/dagger-rooms-conv-threat): after 3 DAgger rounds 12 strict / 16 loose / 28 no contact / 30 mover contact of 60 (base_final 9 / 14 / 15 / 45). Best moving base so far
- [x] Filter alone vs threat alone (3 Nov, r3 strict/loose/no contact/mover contact of 60): filter 12/15/25/35, threat 0/5/25/30, both 12/16/28/30, neither (3 scans) 2/8/23/36. Filter gives the parking, threat the fewer mover hits; both together get both. One seed
- [ ] **Mixture output** (`--mixture 3`, runs/dagger-rooms-mix): several candidate wheel commands, follows the most likely, so left/right is not averaged. Queued after rooms-conv
- [x] **72-ray lidar setting (A3)** (3 Nov): config `lidar_rays` / `--lidar-rays N` on collect_obstacles, dagger, eval_obstacles; ray count read from the data and saved in model.pt; ray filter keeps the same angles (72 rays: window 9, 6 rays per sector, still 48 numbers). 36 stays the default, old models identical row for row. Smoke: 3 expert room runs at 72 rays all clean, 2-epoch model trains and drives. Cost: 1.03 -> 1.12 ms per closed-loop step. No 72-ray dataset yet (waits for the new teacher)
- [ ] **Evade/target switch (A2)** (`--modes --mode-label collision|expert`, built 3 Nov): two wheel heads + learned switch, hysteresis 0.6/0.4 while driving. **Offline gate failed** (runs/modes-gate): catches 53% of evade rows with 42% false alarms at 0.5 (bar: >75%, <30%). Not to be used as it is; retry with 72-ray data or the new teacher's own evade label
- [ ] **Steadier RRT expert** (`SteadyRRTExpert`, `--steady`): near-ties settled by side; new data runs/rooms-expert-steady(-val), then runs/dagger-rooms-steady. Queued after rooms-threat (overnight)
- [ ] **New arm scenarios** (Akhilesh, 3 Nov: later): test the arm on scenario types it has not seen
- [ ] **Generalisation test**: unseen room sizes and layouts, different mover speeds, open floor.
- [?] **Learned position**: optional test in one fixed asymmetric layout. Full SLAM dropped (faculty steer, 2 Nov).

## 4. Evaluation (after the design is frozen)
- [x] **Base metric "parked"** (3 Nov, Akhilesh): reach area (measured, runs/reach-area-fine) + facing +-30 deg + still 2 s + no contact; strict/loose no longer reported. All room runs, the room expert and robust-hard-r3 rescored (rescored.json; originals kept; log runs/rescore-parked-3nov.log)
- [ ] Rescore runs that finish later (mix r3, steady r2-r3) with `python -m comp0245.evaluation.rescore FOLDER`
- [ ] Arm: retrain/DAgger with parking from the learned base and spots spread over the reach area (decided 3 Nov)
- [x] Full robot under adversities (30 Oct, runs/robust-full): clean 37/40, lidar 38, push 35 (expert 35), delay 38, hard 22/30; all contacts while driving. Redo the arm part if model B replaces the arm
- [ ] **Noise and robustness on the new room model** (Akhilesh, 3 Nov): once the model is picked, lidar noise/dropout, push and delay tests (comp0245/evaluation/robust_full.py style) in rooms; noise-robust training if needed (noisy_copy must handle 3 scans and 72 rays first)
- [ ] Bigger test set: generate it BEFORE any test run (current obstacle test split has only 10 scenarios)
- [ ] One final test evaluation: fully learned, expert wheels + arm, full-task expert
- [x] Efficiency (runs/efficiency.json): 7.2k + 6.2k parameters, ~30 kB each, act() median 0.12 ms per 1 ms step
- [ ] Comparisons: ridge baseline, plain BC (round 0) vs DAgger, velocity vs no velocity, model A vs B
- [x] Moving base: velocity vs no velocity with DAgger (3 Nov): velocity 0 strict / 2 loose after 3 rounds, without 3 / 9. Leave it out
- [ ] Moving base: bigger network than 2 x 64, as its own test after scans
- [x] Moving base: 1 vs 3 scans with plain cloning (3 Nov, runs/scans-rooms): 3 scans no fewer mover contacts, parks worse; 2 seeds
- [x] Moving base: 1 vs 3 scans with DAgger (3 Nov, 1 seed): 3 / 9 vs 2 / 8 strict / loose, mover contact 37 vs 36. Still a tie
- [ ] Moving base: noise in room DAgger. `noisy_copy` only handles 1 scan, so 3 scans need that fixed first

## 5. Demo and repo
- [ ] Rehearse the lab demo on the laptop that will be used (clone, RoboEnv, torch, comp0245/apps/demo.py)
- [ ] Final push: code, weights, README updated for whatever is added above. Not on GitHub yet: comp0245/apps/demo.py adversity
      options, DEMO.md, CHECKLIST.md, ROADMAP.md, comp0245/evaluation/robust_full.py, comp0245/evaluation/efficiency.py, comp0245/evaluation/eval_arm.py perturb, val_e and hard3 scenarios
- [ ] README: external sources, and which parts were modified or written from scratch (the brief requires this)

## 6. Report (max 6 pages, IEEE LaTeX) + contribution statement (max 1 page)
- [ ] Introduction, methodology, results, conclusion (docs/COURSE_MAPPING.md links choices to lectures)
- [ ] Figures: replays, DAgger curves, robustness table, failure cases
- [ ] Contribution statement: who wrote what, incl. AI assistance (docs/sources.md)
