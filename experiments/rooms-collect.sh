#!/bin/bash
# 2 Nov: moving-base data in mixed-size rooms (plan: AGENT_LOG.md 13:05 entry). Scenario files, then the visible-only
# RRT expert on train and validation. The test file is only written here; it is not run until the final evaluation.
cd /home/akhjd/ros2_ws/comp0245_project
P=/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python
T=${1:?folder for the part files}
W=${WORKERS:-10}
gen() { $P -B make_obstacle_scenarios.py --out $T/$1.json --seed $2 --prefix $3 --room-half 3.5 7 "${@:4}"; }
MV="--moving-count 2 3"
MX="--moving-count 1 2 --static-count 1 3"
mkdir -p $T
gen train_mv 2101 rmv $MV --train 100 --validation 0 --test 0 &
gen train_mx 2102 rmx $MX --train 100 --validation 0 --test 0 &
gen val_mv 2103 rmv $MV --train 0 --validation 30 --test 0 &
gen val_mx 2104 rmx $MX --train 0 --validation 30 --test 0 &
gen test_mv 2105 rmv $MV --train 0 --validation 0 --test 40 &
gen test_mx 2106 rmx $MX --train 0 --validation 0 --test 40 &
wait
$P -B - "$T" <<'PY' || exit 1
import json, sys
for name in ["train", "val", "test"]:
    rows = sum((json.load(open(f"{sys.argv[1]}/{name}_{k}.json")) for k in ["mv", "mx"]), [])
    json.dump(rows, open(f"scenarios_rooms_{name}.json", "x"), indent=1)
PY
nice -n 5 $P -B collect_obstacles.py --scenarios scenarios_rooms_train.json --out runs/rooms-expert --rrt --duration 110 --workers $W > runs/rooms-expert.log 2>&1 && \
nice -n 5 $P -B collect_obstacles.py --scenarios scenarios_rooms_val.json --out runs/rooms-expert-val --rrt --duration 110 --workers $W > runs/rooms-expert-val.log 2>&1
tail -1 runs/rooms-expert.log
tail -1 runs/rooms-expert-val.log
du -sh runs/rooms-expert runs/rooms-expert-val
