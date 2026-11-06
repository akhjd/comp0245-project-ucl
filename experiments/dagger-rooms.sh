#!/bin/bash
# Moving-base DAgger with a shadow VisibleRRTExpert: 1 scan and 3 scans side by side, then velocity kept in. Test split unused.
cd /home/akhjd/ros2_ws/comp0245_project
export MPLCONFIGDIR=runs/.matplotlib
PY=/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python
COMMON="--data runs/rooms-expert --validation-data runs/rooms-expert-val --scenarios scenarios_rooms_train.json scenarios_rooms_val.json --label-expert rrt_visible --rounds 3 --per-round 100 --duration 70"
nice -n 5 timeout 5h $PY -B dagger.py --name rooms-1scan --no-velocity --start runs/scans-rooms/scans1-seed0 --workers 5 $COMMON > runs/dagger-rooms-1scan.log 2>&1 &
nice -n 5 timeout 5h $PY -B dagger.py --name rooms-3scans --no-velocity --scans 3 --start runs/scans-rooms/scans3-seed0 --workers 5 $COMMON > runs/dagger-rooms-3scans.log 2>&1 &
wait
nice -n 5 timeout 5h $PY -B dagger.py --name rooms-velocity --workers 10 $COMMON > runs/dagger-rooms-velocity.log 2>&1
echo QUEUE FINISHED
