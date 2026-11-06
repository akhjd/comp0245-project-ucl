#!/bin/bash
# 4 Nov: crossing DAgger from window r3 with the arm folded (AGENT_LOG 12:50). One run at a time (16 GB machine).
cd /home/akhjd/ros2_ws/comp0245_project
PY=/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python
nice -n 5 timeout 6h $PY -B dagger.py --name rooms-crossing --start runs/window-fold-r0-start \
    --data runs/rooms-window72 --validation-data runs/rooms-window72-val --lidar-rays 72 \
    --extra-dagger runs/dagger-rooms-window72b/r2-data runs/dagger-rooms-window72b/r3-data \
    --scenarios scenarios_rooms_train.json scenarios_rooms_val.json --label-expert window \
    --hard-scenarios runs/crossing/scenarios_crossing_train.json --hard-duration 70 \
    --rounds 3 --per-round 100 --duration 70 --no-velocity --scans 3 --encoder conv --threat --arm-fold --workers 4 \
    > runs/dagger-rooms-crossing.log 2>&1
echo "CROSSING DAGGER FINISHED $(date)"
