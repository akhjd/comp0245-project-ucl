#!/bin/bash
# 4 Nov: second seed of the 3-round crossing DAgger from window r3 (seed 1), after round 4 (PID $1) finishes.
# 8 simulation workers (about 0.75 GB each; training runs after the workers close). Replaces the second half of
# run_more.sh, which used 4.
cd /home/akhjd/ros2_ws/comp0245_project
PY=/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python
WAIT=$1
if [ -n "$WAIT" ]; then while kill -0 "$WAIT" 2>/dev/null; do sleep 30; done; fi
echo "ROUND 4 FINISHED $(date)"
nice -n 5 timeout 6h $PY -B dagger.py --name rooms-crossing-s1 --start runs/window-fold-r0-start --seed 1 --rounds 3 \
    --extra-dagger runs/dagger-rooms-window72b/r2-data runs/dagger-rooms-window72b/r3-data \
    --data runs/rooms-window72 --validation-data runs/rooms-window72-val --lidar-rays 72 \
    --scenarios scenarios_rooms_train.json scenarios_rooms_val.json --label-expert window \
    --hard-scenarios runs/crossing/scenarios_crossing_train.json --hard-duration 70 \
    --per-round 100 --duration 70 --no-velocity --scans 3 --encoder conv --threat --arm-fold --workers 8 \
    > runs/dagger-rooms-crossing-s1.log 2>&1
echo "SECOND SEED FINISHED $(date)"
