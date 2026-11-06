#!/bin/bash
# 5 Nov overnight (AGENT_LOG 00:18 plan): the crossing DAgger of 4 Nov (runs/crossing/run.sh) with 600 more training
# rooms (scenarios_rooms_train2.json) and their expert data, the mirror flip, early stopping (patience 8) and only the
# last round validated; 12 simulation workers. Seed 0, then seed 1. Rule (before results): adopted only if BOTH seeds
# park more than their 4 Nov versions (35 and 32) with no more mover hits (12 and 13).
# Start only when Akhilesh says so. Steps already done are skipped, so it can be restarted.
ROOT=/home/akhjd/ros2_ws/comp0245_project
cd $ROOT
PY=/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python
O=runs/overnight-rooms
W=${WORKERS:-12}

# 1. Crossing-mover variants of the new rooms; the crossing pool = old + new; an arm-folded copy of the new rooms
[ -f runs/crossing/scenarios_crossing_train2.json ] || \
    $PY -B runs/crossing/make_crossing.py scenarios_rooms_train2.json runs/crossing/scenarios_crossing_train2.json 20261105 || exit 1
[ -f $O/scenarios_crossing_all.json ] || $PY -B - <<'PY' || exit 1
import json
from eval_obstacles import ARM_FOLD
old, new = (json.load(open(f"runs/crossing/scenarios_crossing_train{k}.json")) for k in ("", "2"))
json.dump(old + new, open("runs/overnight-rooms/scenarios_crossing_all.json", "x"), indent=1)
rooms = json.load(open("scenarios_rooms_train2.json"))
json.dump([{**s, "arm_q0": ARM_FOLD} for s in rooms], open("runs/overnight-rooms/scenarios_rooms_train2_fold.json", "x"), indent=1)
print(len(old), "+", len(new), "crossing scenarios;", len(rooms), "new rooms (arm folded copy)")
PY

# 2. The window expert on the 600 new rooms (arm folded, as the models drive)
[ -f runs/rooms-window72-new/manifest.json ] || { nice -n 5 $PY -B collect_obstacles.py \
    --scenarios $O/scenarios_rooms_train2_fold.json --out runs/rooms-window72-new --teacher window \
    --lidar-rays 72 --duration 110 --workers $W > runs/rooms-window72-new.log 2>&1 || exit 1; }
echo "EXPERT DATA FINISHED $(date)"; tail -1 runs/rooms-window72-new.log

# 3. Crossing DAgger on all 800 rooms, two seeds
for S in 0 1; do
    [ -f runs/dagger-rooms-big-s$S/r3/model.pt ] && continue
    nice -n 5 timeout 5h $PY -B dagger.py --name rooms-big-s$S --start runs/window-fold-r0-start --seed $S --rounds 3 \
        --data runs/rooms-window72 runs/rooms-window72-new --validation-data runs/rooms-window72-val --lidar-rays 72 \
        --extra-dagger runs/dagger-rooms-window72b/r2-data runs/dagger-rooms-window72b/r3-data \
        --scenarios scenarios_rooms_train.json scenarios_rooms_train2.json scenarios_rooms_val.json --label-expert window \
        --hard-scenarios $O/scenarios_crossing_all.json --hard-duration 70 \
        --per-round 100 --duration 70 --no-velocity --scans 3 --encoder conv --threat --arm-fold --workers $W \
        --mirror --patience 8 --validate-last > runs/dagger-rooms-big-s$S.log 2>&1
    echo "SEED $S FINISHED $(date)"; grep "ROUND" runs/dagger-rooms-big-s$S.log | tail -1
done
echo "OVERNIGHT FINISHED $(date)"
