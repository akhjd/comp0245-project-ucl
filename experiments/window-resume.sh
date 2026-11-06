#!/bin/bash
# 4 Nov morning: resume the dynamic window DAgger (72 rays) after the 01:40 out-of-memory crash. Round 0 is the
# finished one from runs/dagger-rooms-window72 (via runs/window72-r0-start); rounds 1-3 run here. ONE run only.
cd /home/akhjd/ros2_ws/comp0245_project
PY=/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python
nice -n 5 timeout 7h $PY -B dagger.py --name rooms-window72b --start runs/window72-r0-start \
    --data runs/rooms-window72 --validation-data runs/rooms-window72-val --lidar-rays 72 \
    --scenarios scenarios_rooms_train.json scenarios_rooms_val.json --label-expert window --rounds 3 --per-round 100 \
    --duration 70 --no-velocity --scans 3 --encoder conv --threat --workers 4 > runs/dagger-rooms-window72b.log 2>&1
echo "WINDOW RESUME FINISHED $(date)"
