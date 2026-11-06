# Roadmap to 6 Nov

Updated Mon 2 Nov. Julia's `rrt-expert` work is merged locally; decisions from the 2 Nov discussion are in. A plan
for Akhilesh to change. Status of each item stays in CHECKLIST.md. Times come from
`runs/overnight.log` unless marked as an estimate.

## Decisions so far
Quick reference, newest thinking only. The reasons are in AGENT_LOG.md.

| Topic | Decision | Still to do |
|---|---|---|
| Final open-floor robot | Stays as it is: no velocity, 42 inputs, 2 x 64 (7k parameters), 37/40. Rechecked 2 Nov, unchanged | Nothing |
| Two versions | Julia fixes her MLP; we build up from our 7k model. Same simulator, expert and validation sets | Pick one on Tue evening |
| Velocity | Dropped for good (Akhilesh, 3 Nov); later room runs are 1 or 3 scans without it. Room DAgger 3 Nov: with velocity 0 strict / 2 loose after 3 rounds, without 3 / 9 | Nothing (comparison table for the report) |
| Network size | Akhilesh wants to go above 7k. RRT data: 3 x 256 memorised (R² 0.87 train, 0.06 new runs). **Rechecked 4 Nov on the window teacher** (`size_check.py`, 2 seeds): validation copy near movers 64: 0.65, 128: 0.70, 256: 0.66 while train rises 0.80 / 0.84 / 0.88; rule needed +0.1, so size is not the fix. Gap is train to new rooms (best epoch 1-2) | Nothing |
| Mover perception (window teacher) | Cheat test 4 Nov: giving the model the movers' true position and velocity raises copying near movers only 0.55 -> 0.58 (rule needed +0.1). Not the bottleneck | Suspects left: room variety, thin 0.25 m margin |
| Arm while the base drives | **Folded inside the base outline** (Akhilesh 4 Nov; pose from `runs/arm-tuck/search.py`, scenario `arm_q0`). The old pose reached 0.73 m ahead at 0.45-0.54 m high, below the 0.6 m obstacle tops. Window r3 with the fold: 30 parked / 16 mover hits (was 20 / 23) | Unfold to the usual start pose before the arm works; rescore other models with the fold for fair comparison |
| Output type (hedging) | Checked 4 Nov (`output_check.py`): mixture, menu (63 options + hold), evade weighting all fail the rule. On the expert's own states regression already turns like the expert (0.80); in the 4 s before real hits every variant is poor (0.05-0.12). Not adopted | Hits are in states the expert never visits: targeted crossing-mover DAgger data is the lead |
| Crossing DAgger (room model) | **Adopted 4 Nov**: `runs/dagger-rooms-crossing` r3 (from window r3, arm folded; each round 50 normal + 50 crossing-mover runs, expert-labelled). Parked 30 / 27 / 35 / 35 of 60, mover hits 16 / 27 / 18 / 12; still-obstacle hits 0. Copy score in the 4 s before r3's real hits 0.21 -> 0.55. Passed the pre-written gate (beat 30 parked with <= 16 mover hits). **Second seed** (`dagger-rooms-crossing-s1`): r3 32 parked / 13 mover hits, copy score 0.53. Fewer mover hits (16 -> 12-13) and better copying repeat on both seeds; the parking gain only partly (+4 vs +1; 19 of 60 rooms differ between the seeds). Round 4 (`dagger-rooms-crossing-r4`): 35 / 14, not kept | Test split only at the final evaluation |
| Parking rule and hold (4 Nov) | **Measured reach** adopted (Akhilesh): reach limit 84.9% of the straight arm facing the target, down to 83% at 30 deg (measured edges); every room model rescored (rescored-reach.json, 0 to +2 each; 83% column kept). **Latched hold** adopted: learned "expert says exactly 0" head on frozen crossing r3, holds once on for 0.5 s; r3 + latched hold parks 37 / 60 (35 on the 83% rule), mover hits 12, loses no room vs r3. Hold head failed its offline rule. **Second seed with the hold: 32 -> 31** (fixed the creeping room, but held still ~14 cm outside the reach area in 2 rooms), so the hold's gain does not repeat; it stays in the adopted model because the pre-written rule picked it, reported as no clear gain. 100 s check (report-only): 39 by 100 s vs 37 by 70 s | Teacher parking deeper not run (would target the stop-short holds); final test of the adopted model |
| More rooms + mirror (5 Nov overnight) | **Done, not adopted.** 600 new training rooms (`scenarios_rooms_train2.json`) + expert data, mirror flip, early stopping, crossing DAgger x 2 seeds (`runs/dagger-rooms-big-s0`, `-s1`). Round 3: seed 0 28 parked / 15 mover hits, seed 1 36 / 15. Rule was both seeds > 35 / 32 with <= 12 / 13 hits: fails (mover hits worse on both). Seed 0 r3 + latched hold (37 / 12) stays the room model. Speed-ups measured: GPU no faster, batch 4096 copied worse; 100 Hz ruled out. ChatGPT hand edits: none beat the parent | Done 5 Nov: arm hand-off fixes (rays/scans from the model, parking rule for 2 s before the arm, unfold the folded arm). Full robot on 60 validation rooms: **32 / 60** (16 lost to contact while driving, 6 never parked, 6 in the arm phase). **Final test (80 rooms, once): base 48 / 80 parked (60%), 18 mover hits; full robot 45 / 80 (56%)**, both in line with validation. **New arm for the room robot (5 Nov, adopted):** retrained with 1,000 expert reaches from where the room base parks (`runs/arm-parked/arm`); arm-only on the 49 validation parking spots 38 vs 25, full robot 37 / 60 vs 32 / 60, no room lost. **Then arm DAgger on parked spots, 2 rounds (adopted round 2, `runs/arm-parked/dagger-r2/arm`, copied to `weights/arm_rooms.pt`):** 49 spots 45, 200 fresh synthetic spots 188 (was 103), full robot 37 / 60 (same rooms). Test not rerun: test numbers stay the old arm's. Next: the report |
| Scans | A tie so far. Plain cloning: 3 scans parked worse. Room DAgger r3 (1 seed): 1 scan 3 strict / 9 loose / 37 mover contact, 3 scans 2 / 8 / 36. Neither touches movers less | Only worth keeping if something makes the network use motion (threat prediction) |
| Room DAgger | Works and helps parking (1 scan loose 3 -> 9 over 3 rounds) but alone not mover contact (36-45 of 60 runs) | Nothing |
| Ray filter + threat prediction | **Best moving base so far** (3 Nov, `runs/dagger-rooms-conv-threat`, 3 scans, 8.6k parameters): r3 12 strict / 16 loose / 28 no contact / 30 mover contact of 60, against `base_final` 9 / 14 / 15 / 45 and plain 3 scans 2 / 8 / 23 / 36. One seed | Filter alone r3 12/15/25/35, threat alone 0/5/25/30: the filter brings the parking, threat prediction the fewer mover hits. Keep both |
| Averaging left and right | Akhilesh 3 Nov: "Both, mixture first". (1) Mixture output: 3 candidate commands with probabilities, the base follows the most likely and sticks with it until another is twice as likely. Diffusion Policy rejected: too slow for 1 ms decisions and needs more data. (2) `SteadyRRTExpert`: near-ties (within 10%) go to the same side as its last plan. First check: the mixture rarely keeps opposite turns on saved runs, so averaging may be a smaller cause than thought | Driving scores decide: runs/dagger-rooms-mix, runs/dagger-rooms-steady (overnight) |
| Lidar rays (A3) | Akhilesh 3 Nov: 72 rays (5 deg) for new runs, 36 stays the default and for every existing model. Built: `--lidar-rays N`, ray count travels with the data and the model; ray filter keeps the same angles (window about 40 deg, 30 deg sectors, 48 numbers). Cost about 9% per step (1.03 -> 1.12 ms) | Record 72-ray data once the new teacher is chosen |
| Evade/target switch (A2) | Built 3 Nov (`--modes`): target and evade wheel heads, learned switch, hysteresis enter 0.6 / leave 0.4. Evade label "collision": robot and mover at constant velocity come within 0.47 m + mover radius + 0.3 m in 3 s (23% of room validation rows). **Gate failed**: 53% caught, 42% false alarms (bar 75% / 30%); a switch trained alone does no better (best 54% / 49%), with or without velocity; the mover is hit by a ray on 99.97% of evade rows | Do not use as it is. Retry on 72-ray data, or with the new teacher's own evade label (`--mode-label expert`) |
| Base metric | **Parked** (Akhilesh 3 Nov): the base stops where the arm can reach the target (horizontal distance 0.3 m up to 83% of the straight arm at the target's height, 0.98-1.02 m in the rooms; measured with the expert arm in runs/reach-area and runs/reach-area-fine), facing it within +-30 deg, still for the last 2 s, no contact all run (a mover hitting the parked robot counts). Strict/loose dropped from reporting: off the expert's route the best stopping place differs, so matching the expert's ring is not required. Old and rescored results both kept (rescored.json next to evaluation.json) | Report parked + mover hits; strict/loose only as "dropped, because". Arm training from now on uses parking from the learned base (plus spots spread over the reach area), not the base expert |
| Teacher | **Dynamic window teacher** (`WindowExpert`, 3-4 Nov) won the teacher comparison: 16/20 clean, copy R² 0.46 vs the RRT's 0.06 on the same rooms. Window DAgger (`runs/dagger-rooms-window72b`, 72 rays, filter + threat) parked 9 / 16 / 17 / 20 of 60, mover hits 42 / 30 / 24 / 23 (fewest of any learned room model); steady r3 22 / 27, 1 seed each, so a tie on parking | Decide the room model; second seeds |
| Julia's data | Received 3 Nov: `runs/julia-dataset`, 1,650 runs (1,625 successful), full-knowledge expert, 656 with movers, 8 x 8 and 10 x 10 m rooms | Needs a manifest for our loader; train split only; test with and without the mover runs |
| A* | Inside the expert only, never while the model drives | Nothing |
| SLAM, particle filter | Dropped: faculty prefer learning over built-in code | Drift test only |
| Rooms | Walls stay; rooms 7-14 m wide, so walls are sometimes out of lidar range | Nothing |
| Model B | **Not kept** (3 Nov, Claude took it over): 42/90 against plain A's 62/90, same data and rules, 1 seed. Offline error slightly better. 32 of its 48 misses came within 3 cm and then moved away After both DAgger stages too: 69/90 fully learned and 37/40 on val_d, against the final arm's 77/90 and 37/40 | Report it as tried (with DAgger); the arm stays `weights/arm_final.pt` |
| Split direction and distance | An idea for model B: 7 directions with their size removed, plus distance | Dropped: B did not stop short, it got close and left || Goal arm pose | Not doing it: a PD controller could then do the job, the expert's data does not match it, and our misses are not at joint limits | Report as future work |
| Merged | Julia's 00984f1 is in this folder. Her 19237ea is not, and her `settings.json` must not be taken. Nothing is on GitHub `main` yet | Merge her DAgger code; pushing needs Akhilesh's yes |
| Test split | Runs once, after the freeze | Wed night |
| Combined room model (6 Nov) | One model for our rooms + Julia's: 0.8 m/s window teacher, lidar 144 rays / 10 m, no A* at run time, 3 DAgger rounds, recipe unchanged (Akhilesh: "keep it as is"). Rule: ours >= 35 parked with <= 13 mover hits AND hers >= current + 10. Disk filled at 01:51; resumed 07:31 | Result ~13:30-14:30 |
| GPU rerun, batch 1024 (6 Nov) | After the combined run, the same DAgger on the same teacher data with training on the GPU (rocm-test venv, used as is) at batch 1024, while Akhilesh writes the report | Build after the combined run ends (no edits to dagger.py / train_base_torch.py mid-run) |

## The fixed dates
- **Fri 6 Nov:** report, contribution statement and code due. (Julia's notes say "about 8 Nov". The brief says 6 Nov:
  check this with her.)
- **Lab demo, by 6 Nov:** in person, on a laptop. Not marked, but required. It need not be final.
- **Wed 4 Nov, 18:00 (proposed): feature freeze.** The test evaluation runs that night, once.

## Where things stand
Most of the list now exists as an **expert**. Almost none of it exists as a **learned model**, and the course asks
for behaviour cloning. That gap is now the whole project.

| Item | Expert | Learned |
|---|---|---|
| Open floor with obstacles | Done | **Done:** 37/40, expert 40/40 |
| Bounded room | Done (Julia) | 30/40 with no retraining |
| Moving cylinders | Done (Julia, privileged RRT); visible-only teacher 183/200 in mixed-size rooms | **0/10** moving only, **2/10** mixed, no retraining |
| Floor plans + A* | Done (Julia) | **1/20**, no retraining |
| Several dots, order, arm return | Done (Julia) | Not possible yet: arm return only exists in the expert |
| Model B | Not relevant | Not started (Akhilesh) |
| Edge cases, very fast obstacles | Not started | Not started |
| Red balls, initial position, multiple robots | Not started | Not started |

### What was checked on her branch (2 Nov, scratch copy, test split untouched)
- **Merge:** no conflicts. 15 unit tests pass. One episode, learned and expert, stays identical row for row.
- **Her expert:** 20/20 on her own moving and mixed scenarios. **18/20 on 20 fresh floor plans** I generated, against
  her 30/30. Both failures involved a mover, and in one the robot touched a mover and a wall over 11 s. Only 6 of the
  20 had a mover, so it needs a bigger test, but mover contact in tight spots is the expert's real weakness.
- **Her speed:** a plan takes 0.6-2 s on average, 4.6 s at worst, and episodes run at 1.0-1.75 x sim time. Fast enough.
- **Moving runs failed by driving into the movers at full speed.** Expected: one lidar scan cannot show motion.
- **Room runs failed after parking.** All 10 parked and stopped, then the arm missed. Side offsets reached 16 cm against
  a success median of 2 cm. This looks like the known base-to-arm shift, and may be what model B helps with.

### Changes to last night's roadmap
- **RRT recommendation withdrawn.** It is built and it works, so keep it as the teacher. The concern behind it remains
  and moves to the learning step: the expert acts from a stored plan the network cannot see.
- **Several dots:** the order is hers: it finishes a room before leaving, rather than the shortest route.
- **A*:** stays inside the expert, guiding RRT through doorways. Decided 2 Nov: no A* while the model drives.

## The plan: three learned results, in order
Do not replace the current base. It stays the open-floor model, and its results stand. Everything new is a second
model, so a bad result here cannot damage what already works.

### 1. Merge (done 2 Nov)
Her 9 files are in the project folder. GitHub is a separate step and needs Akhilesh's permission. **Agree with Julia who
does what**, because her Phase 4 is this same work.

### 2. Moving base (the main result)
This answers the brief's "single instant vs temporal interval" on a task where the answer should matter.
1. **Visible-only expert.** A mover counts only if a lidar ray hit it in the last 1 s, with constant-velocity prediction
   and no turning points. Walls, furniture and static cylinders stay known. **Passed 2 Nov and is now the teacher:** 117/120 against
   119/120 for the full expert on the same scenarios, gate 90% (`compare_experts.py`). In the mixed-size rooms it scores
   lower: 183/200 and 54/60 (see step 3).
2. **Done 2 Nov.** Train, validation and test sets with movers, in **rooms of mixed size** (7 to 14 m wide, each layout
   moved to a random place in its room): `scenarios_rooms_{train,val,test}.json`. 200, 60 and 80 scenarios, half moving
   only and half mixed. The test file has not been run.
3. **Done 2 Nov.** Expert data: `runs/rooms-expert` (183/200 clean, 6.99 M rows) and `runs/rooms-expert-val` (54/60).
   No wall is in lidar range on about 15% of training rows. Movers are hit by a lidar ray on 90% of rows.
   **Weakness to watch:** of the 23 failures, the full expert solves 19. 18 involve a mover, none of which reversed in the
   last 3 s and only one of which was out of sight, but those runs had 16-126 failed plans. So the expert sees the mover
   and cannot find a way round in time. Likely but unproven: late warning, and movers can outrun the robot (0.3 against
   0.25 m/s). Failed episodes are left out of training, but DAgger will ask the expert about exactly these situations.
4. **Done 2-3 Nov, plain cloning only.** 1 scan against 3 scans spread over about 1 s (114 inputs, about 11.8k parameters).
   0.2 s is too short: a cylinder at 0.2 m/s moves 4 cm while rays are about 35 cm apart at 2 m. Same data, no velocity,
   2 x 64, 2 seeds, arm held, 60 room validation scenarios (`runs/scans-rooms/results.json`). Results below are seed 0 / seed 1.
   | Model | Offline RMSE | Strict | Loose | No contact | Mover contact | Median distance error |
   |---|---|---|---|---|---|---|
   | 1 scan, plain cloning | 1.17 / 1.16 | 1 / 0 | 3 / 3 | 18 / 15 | 36 / 41 | 0.20 / 0.15 m |
   | 3 scans, plain cloning | 1.02 / 1.00 | 0 / 0 | 0 / 1 | 16 / 21 | 42 / 36 | 0.39 / 0.19 m |
   | `weights/base_final.pt` (1 scan, DAgger, never saw rooms) | - | 9 | 14 | 15 | 45 | 0.04 m |
   **Reading:** 3 scans has the better offline error but no fewer mover contacts, and parks worse: "offline better, driving
   worse" again. All 60 runs moved, so this is not the standstill copycat. The final base parks about 4 times closer with
   only 1 scan, which points at DAgger and data, not scans, as the larger gap; it was also trained on more data, so this is
   a pointer and not proof. Every model above hits movers in most runs. 1 vs 3 scans must be repeated with DAgger.
5. **Done 3 Nov: threat prediction and a learned ray filter** (`train_base_torch.py --threat --encoder conv`).
   Threat: a training-only output (will a mover come within 1 m of the robot in the next 3 s; its position and velocity
   in the robot's directions), from the simulator's true positions, ignored while driving. Ray filter: one small set of
   weights looks at 5 neighbouring rays in all 3 scans and slides round all 36 positions; results stay in order, so
   position = direction; pooled to 12 sectors x 4 detectors (48 numbers instead of 108). Room DAgger, same protocol as
   step 6, strict / loose / no contact / mover contact of 60, r0 -> r3: 3/15/29/27, 5/10/23/34, 12/17/27/32, 12/16/28/30.
   Plain cloning alone (r0) already beat the plain 3-scan network (0/0/16/42). Filter alone and threat alone: running.
6. **DAgger with a shadow expert, with noise on the model's inputs only.** Her notes say DAgger cannot use the RRT expert
   yet, because its clock and plan are not updated while the network drives. That is the same problem we solved for the
   arm: keep a shadow expert running beside the policy. It already replans when it strays 0.3 m from its plan, so this
   looks like a small change. **Done 3 Nov, no noise yet** (`runs/dagger-rooms.sh`, `dagger.py --label-expert rrt_visible`):
   the visible-only expert replans beside the model and labels every state; a labelling run stops after 2 s touching
   something or 20 s standing still. 3 rounds x 100 runs on the 200 training rooms, 2 x 64, arm held, scored by driving
   the 60 room validation scenarios, 1 seed. Strict / loose / no contact / mover contact per round r0 -> r3:

   | Model | r0 (plain cloning) | r1 | r2 | r3 |
   |---|---|---|---|---|
   | 1 scan, no velocity | 1 / 3 / 18 / 36 | 0 / 2 / 17 / 37 | 2 / 9 / 18 / 41 | 3 / 9 / 20 / 37 |
   | 3 scans, no velocity | 0 / 0 / 16 / 42 | 0 / 2 / 17 / 43 | 1 / 6 / 18 / 39 | 2 / 8 / 23 / 36 |
   | 1 scan, velocity | 0 / 0 / 11 / 44 | 0 / 0 / 20 / 33 | 0 / 2 / 18 / 40 | 0 / 2 / 14 / 45 |
   | `weights/base_final.pt` | 9 / 14 / 15 / 45 | | | |

   **Reading:** DAgger helps parking (median miss 0.20 -> 0.08 m for 1 scan) but not mover contact, which stays at 36-45
   of 60 runs in every model, as it does for the final base. Velocity loses: copycat at r0 (median miss 2.7 m) and still
   worst after 3 rounds. 1 vs 3 scans is still a tie. 63% of labelling runs ended by crashing, so the expert's answers near
   movers mostly come from the 2 s before contact. Offline error rises each round (DAgger adds harder states), as expected.
7. Then the drift test (success against position error) and the generalisation test (unseen room sizes and layouts,
   different mover speeds, and open floor).

**Stop rule:** if DAgger cannot be made to work by Tue evening, report plain BC honestly. The gap between a privileged
teacher and a lidar student is a result in itself, and she has already named it as a measured risk.

### 3. Arm in the new worlds
Needed either way: the room failures are arm misses. Run arm DAgger with the moving base parking (about 1 h), and run
model B against the 10 room failures as well as the 90 arm scenarios. Off-side parking is what direction inputs
should help with, so this is B's strongest test.

### 4. Floor plans (only after 2 and 3)
Lidar and target only, no A* at run time (decided 2 Nov). Train on floor plans with the A*-guided expert as teacher. The model may learn to head for a door it can see, but has nothing to go on when the door is out of range, so expect a partial result and report it. Learned several-dots needs the arm-return logic moved out of `expert_action` so it can run
alongside a policy, which is small, and is what would make red balls possible.

## Everything else on the list
| Item | Tier | Decision | Estimate | Stop if |
|---|---|---|---|---|
| Model B | 1 | Build (Akhilesh) | One evening + 15 min compare | Does not beat plain A (62/90 expert parking): keep A, report it. To replace the final arm it must pass 87/90 and 77/90 after arm DAgger |
| Comparisons (ridge, BC vs DAgger, velocity, A vs B) | 1 | Build | 2 h, mostly existing runs | Never skip |
| Edge cases | 2 | Test on expert and learned | Half a day | Never skip |
| Very fast obstacles | 2 | Test only: success against speed, 0.1 to 2 m/s. Her expert is only tested up to 0.3 m/s | 20 min | None |
| Expert size and speed sweep | 2 | In her optional list; merge with the sweep above | 20 min | None |
| Red balls into a box | 3 | Demo, once learned several-dots works | Half a day | Demo only |
| Find initial position | 3 | Particle filter dropped (faculty prefer learned over built-in code). Drift test on the learned robot; learned position only as a test in one fixed asymmetric layout | 20 min; half a day for the optional test | Optional test: drop at the Wed freeze |
| Multiple robots, different goals | 3 | Build if moving base works | 1 day | Wed freeze |
| Multiple robots, same goal | 3 | Only if time is left | Half a day | Wed freeze |
| Robot picking itself back up | - | **Recommend fall detection instead.** Rough estimate: about 150 N m needed, arm capped at 30 N m | 1 h | - |
| Full SLAM | - | **Dropped** (faculty steer, 2 Nov) | - | - |

The report's 6 pages hold about two or three extensions. Moving base + teacher-student + temporal input is the first.
Floor plans, and what a memoryless lidar policy cannot do there, could be the second. The rest is demo or future work.

## Day by day
| Day | Akhilesh | Julia (to agree) | Claude (with go-ahead) | Overnight |
|---|---|---|---|---|
| **Mon 2** | `ik_direction`, test, train B; agree roles with Julia | Share fair-test sets; fix mover contact in tight spots | Merge, scenario sets, expert data (all done); speedups, stacked-scan inputs, shadow-expert change | BC on 1 vs 3 scans |
| **Tue 3** | Model B verdict; report methods | Expert tests; expert section of the report | Moving DAgger, arm DAgger, room recheck | Moving DAgger rounds, arm DAgger |
| **Wed 4** | Decide what makes the cut | Experiments or report | Floor-plan trial, edge cases, speed sweep, extras. Generate test sets | **18:00 freeze.** Final test, robustness, comparisons, efficiency |
| **Thu 5** | Results, conclusion, contribution | Report, contribution | Figures, README, clean-clone check | Spare |
| **Fri 6** | Final read; submit | Final read | Final fixes | |

If a day slips, the extras give way first, never the freeze. Report text on the problem, base, arm, DAgger and
robustness can start today, because those results already exist. Julia's `report_progress.txt` has decisions and
reasons ready for the methodology section.

## Evaluation rules to keep
- Test sets are generated before the freeze, for the open floor and each kept environment. Each runs once.
- The moving model is a second model, so the open-floor model needs no swap test. If it were ever to replace it, it
  would have to pass `val_e` and `hard3` first.
- Rerun `robust_full.py` on the final open-floor robot (22 min) if model B replaces the arm.
- Report the expert beside the learned robot, failures included.
- Expert results need their scenario sets in the repo, or they cannot be reproduced.

## Do first, whatever else changes
1. **Roles:** Akhilesh and Julia agree who owns Phase 4, so it is not built twice.
2. **Lab demo:** attend the earliest session you can, with the current system.
3. **Laptop:** check RoboEnv, torch and `demo.py` run on it, soon.
4. **Speedups:** skip the unused mass matrix and use 10-12 workers. Ten workers ran fine today.
5. **README:** external sources, and which parts were modified or written from scratch. `--rrt`, `--room` and floor
   plans are not documented yet.

## Open questions for Akhilesh
1. Is the deadline 6 Nov or 8 Nov?
2. Has the lab demo happened or been booked, and which laptop?
3. Does Julia plan to start Phase 4 herself? Who else is on the team?
4. Can she share her fair-test scenario sets or seeds?
5. Do you agree with fall detection?
6. If dot ordering is built, does faculty accept it as a run-time rule?
