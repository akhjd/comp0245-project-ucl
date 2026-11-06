#!/bin/bash
# 3 Nov: room DAgger with the learned ray filter (--encoder conv) and/or the training-only threat output (--threat).
# Same protocol as runs/dagger-rooms.sh (rooms-expert, visible-only RRT shadow, 3 rounds x 100, 60 room validation,
# no velocity, 3 scans), so each compares with runs/dagger-rooms-3scans. Round 0 is trained here (plain cloning).
# Both-together starts now at 4 workers; the other two wait for the model B arm DAgger to free memory.
cd /home/akhjd/ros2_ws/comp0245_project
PY=/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python
COMMON="--data runs/rooms-expert --validation-data runs/rooms-expert-val --scenarios scenarios_rooms_train.json scenarios_rooms_val.json --label-expert rrt_visible --rounds 3 --per-round 100 --duration 70 --no-velocity --scans 3"
nice -n 5 timeout 7h $PY -B dagger.py --name rooms-conv-threat --encoder conv --threat --workers 4 $COMMON > runs/dagger-rooms-conv-threat.log 2>&1 &
while pgrep -f "[d]agger_arm.py --name modelb" > /dev/null; do sleep 60; done
echo "arm DAgger finished $(date)"
nice -n 5 timeout 7h $PY -B dagger.py --name rooms-conv --encoder conv --workers 3 $COMMON > runs/dagger-rooms-conv.log 2>&1 &
nice -n 5 timeout 7h $PY -B dagger.py --name rooms-threat --threat --workers 3 $COMMON > runs/dagger-rooms-threat.log 2>&1 &
wait
echo QUEUE FINISHED
