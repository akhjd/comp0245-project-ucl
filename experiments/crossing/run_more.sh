#!/bin/bash
# 4 Nov (AGENT_LOG 17:06): round 4 of the crossing DAgger from r3 (seed 3: new rooms, all earlier crossing data kept),
# then the second seed of the 3-round crossing DAgger from window r3 (seed 1). One run at a time (16 GB machine).
cd /home/akhjd/ros2_ws/comp0245_project
PY=/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python
WAIT=$1     # optional: PID to wait for first (the hold diagnostic)
if [ -n "$WAIT" ]; then while kill -0 "$WAIT" 2>/dev/null; do sleep 30; done; fi
C=runs/dagger-rooms-crossing
COMMON="--data runs/rooms-window72 --validation-data runs/rooms-window72-val --lidar-rays 72
    --scenarios scenarios_rooms_train.json scenarios_rooms_val.json --label-expert window
    --hard-scenarios runs/crossing/scenarios_crossing_train.json --hard-duration 70
    --per-round 100 --duration 70 --no-velocity --scans 3 --encoder conv --threat --arm-fold --workers 4"
nice -n 5 timeout 3h $PY -B dagger.py --name rooms-crossing-r4 --start runs/crossing-r4-start --seed 3 --rounds 1 \
    --extra-dagger runs/dagger-rooms-window72b/r2-data runs/dagger-rooms-window72b/r3-data \
        $C/r1-data $C/r1-hard-data $C/r2-data $C/r2-hard-data $C/r3-data $C/r3-hard-data \
    $COMMON > runs/dagger-rooms-crossing-r4.log 2>&1
echo "ROUND 4 FINISHED $(date)"
nice -n 5 timeout 6h $PY -B dagger.py --name rooms-crossing-s1 --start runs/window-fold-r0-start --seed 1 --rounds 3 \
    --extra-dagger runs/dagger-rooms-window72b/r2-data runs/dagger-rooms-window72b/r3-data \
    $COMMON > runs/dagger-rooms-crossing-s1.log 2>&1
echo "SECOND SEED FINISHED $(date)"
