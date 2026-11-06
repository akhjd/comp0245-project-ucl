#!/bin/bash
# 3 Nov 12:45: replaces the tail of runs/dagger-rooms-3.sh, whose queue was stopped (the model B arm stage it started
# keeps running). Memory freed early, so the ray-filter-only run starts now; after the arm stage: round choice, the
# two val_d checks, and the threat-only run.
cd /home/akhjd/ros2_ws/comp0245_project
PY=/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python
BASE=runs/dagger-novnoise-hard/r3/model.pt
COMMON="--data runs/rooms-expert --validation-data runs/rooms-expert-val --scenarios scenarios_rooms_train.json scenarios_rooms_val.json --label-expert rrt_visible --rounds 3 --per-round 100 --duration 70 --no-velocity --scans 3"
nice -n 5 timeout 7h $PY -B dagger.py --name rooms-conv --encoder conv --workers 3 $COMMON > runs/dagger-rooms-conv.log 2>&1 &
while pgrep -f "[d]agger_arm.py --name modelb-lb" > /dev/null; do sleep 30; done
echo "arm stage finished $(date)"
nice -n 5 timeout 7h $PY -B dagger.py --name rooms-threat --threat --workers 3 $COMMON > runs/dagger-rooms-threat.log 2>&1 &
K=$($PY -B runs/choose_arm_round.py runs/dagger-arm-modelb-lb | tail -1)
ARM=runs/dagger-arm-modelb-lb/r$K/model.pt
echo "chosen arm $ARM"
F="--fresh 0 40 --no-held-out --fresh-file scenarios_obstacles_val_d.json --workers 3"
nice -n 5 $PY -B eval_arm.py --name final-modelb --arm $ARM --base $BASE $F > runs/arm-eval-final-modelb.log 2>&1
nice -n 5 $PY -B eval_arm.py --name final-modelb-hybrid --arm $ARM $F > runs/arm-eval-final-modelb-hybrid.log 2>&1
echo "arm checks finished $(date)"
wait
echo QUEUE 4 FINISHED
