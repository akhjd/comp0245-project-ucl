# Experiment launchers

The small `.py` / `.sh` scripts that ran the experiments, copied from the local `runs/` folder (6 Nov 2026) with
their folder layout. The data and results they produced are not in Git (`runs/` is about 200 GB); the numbers are
logged in `docs/DATASETS.md` and `docs/ROADMAP.md`. Most scripts hardcode the project path
(`/home/akhjd/ros2_ws/comp0245_project`) and expect to run from inside `runs/`, so read them as a record of what
was run rather than as ready-to-run tools.

- `overnight-combined/`: the 6 Nov combined room model (our rooms + Julia's floor plans, 0.8 m/s window teacher,
  144-ray 10 m lidar, DAgger).
- `crossing/`, `scans-rooms/`, `modes-gate/`: room-model experiments (crossing movers, scan history, output types).
- `arm-parked/`, `arm-tuck/`: the room arm (`weights/arm_rooms.pt`) and arm folding.
- `teacher-speed/`, `teacher-compare/`: expert comparisons.
- `final-test/`: the single final test run.
- `pause.sh` / `resume.sh`: freeze and continue running jobs (for demos).
