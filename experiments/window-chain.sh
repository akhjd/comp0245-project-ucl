#!/bin/bash
# 3-4 Nov overnight: the dynamic window teacher (WindowExpert) won the teacher comparison (16/20 clean, copy R2
# 0.46 vs the RRT's 0.06 on the same 40 + 20 rooms). Record its full room data with 72 rays and, to separate the
# teacher's effect from the rays', also with 36 rays; then room DAgger on each (ray filter + threat, 3 scans, no
# velocity, labels from a shadow WindowExpert), same protocol as dagger-rooms-conv-threat.
cd /home/akhjd/ros2_ws/comp0245_project
PY=/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python
for R in 72 36; do
    nice -n 5 $PY -B collect_obstacles.py --scenarios scenarios_rooms_train.json --out runs/rooms-window$R --teacher window \
        --lidar-rays $R --duration 110 --workers 4 > runs/rooms-window$R.log 2>&1
    nice -n 5 $PY -B collect_obstacles.py --scenarios scenarios_rooms_val.json --out runs/rooms-window$R-val --teacher window \
        --lidar-rays $R --duration 110 --workers 4 > runs/rooms-window$R-val.log 2>&1
    echo "window data $R rays finished $(date)"
done
COMMON="--scenarios scenarios_rooms_train.json scenarios_rooms_val.json --label-expert window --rounds 3 --per-round 100 --duration 70 --no-velocity --scans 3 --encoder conv --threat --workers 3"
for R in 72 36; do
    nice -n 5 timeout 7h $PY -B dagger.py --name rooms-window$R --data runs/rooms-window$R --validation-data runs/rooms-window$R-val \
        --lidar-rays $R $COMMON > runs/dagger-rooms-window$R.log 2>&1 &
done
wait
echo "WINDOW CHAIN FINISHED $(date)"
