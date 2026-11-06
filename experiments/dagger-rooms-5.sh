#!/bin/bash
# 3 Nov 15:50 (Akhilesh: "Both, mixture first"). Same room DAgger protocol as rooms-conv (ray filter, 3 scans, no
# velocity), one change each:
#   rooms-mix:    mixture output (3 candidate wheel commands), starts when rooms-conv finishes (frees its memory)
#   rooms-steady: SteadyRRTExpert data + labels; its expert data is collected when rooms-threat finishes
cd /home/akhjd/ros2_ws/comp0245_project
PY=/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python
SC="--scenarios scenarios_rooms_train.json scenarios_rooms_val.json --rounds 3 --per-round 100 --duration 70 --no-velocity --scans 3 --encoder conv"
while pgrep -f "[n]ame rooms-conv --encoder" > /dev/null; do sleep 30; done
echo "rooms-conv finished $(date)"
nice -n 5 timeout 7h $PY -B dagger.py --name rooms-mix --mixture 3 --workers 3 --data runs/rooms-expert \
    --validation-data runs/rooms-expert-val --label-expert rrt_visible $SC > runs/dagger-rooms-mix.log 2>&1 &
while pgrep -f "[n]ame rooms-threat --threat" > /dev/null; do sleep 30; done
echo "rooms-threat finished $(date)"
nice -n 5 $PY -B collect_obstacles.py --scenarios scenarios_rooms_train.json --out runs/rooms-expert-steady --rrt --steady \
    --duration 110 --workers 4 > runs/rooms-expert-steady.log 2>&1
nice -n 5 $PY -B collect_obstacles.py --scenarios scenarios_rooms_val.json --out runs/rooms-expert-steady-val --rrt --steady \
    --duration 110 --workers 4 > runs/rooms-expert-steady-val.log 2>&1
echo "steady expert data finished $(date)"
nice -n 5 timeout 7h $PY -B dagger.py --name rooms-steady --workers 4 --data runs/rooms-expert-steady \
    --validation-data runs/rooms-expert-steady-val --label-expert rrt_steady $SC > runs/dagger-rooms-steady.log 2>&1
wait
echo QUEUE 5 FINISHED $(date)
