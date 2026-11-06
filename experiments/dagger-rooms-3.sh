#!/bin/bash
# 3 Nov: model B passed model A in arm DAgger with expert parking (88/90 at r5), so it gets the same second stage
# as the final arm (runs/arm-b-pipeline.sh): arm DAgger with the learned base parking, round chosen on the 90, then
# the final full-robot check on scenarios_obstacles_val_d.json (where the final arm scored 37/40 and 40/40).
# Then the two room DAgger runs paused to free memory (ray filter alone, threat alone), as in runs/dagger-rooms-2.sh.
cd /home/akhjd/ros2_ws/comp0245_project
PY=/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python
BASE=runs/dagger-novnoise-hard/r3/model.pt
nice -n 5 $PY -B dagger_arm.py --name modelb-lb --rounds 3 --workers 4 --ik --base $BASE \
    --init-model runs/dagger-arm-modelb/r5/model.pt --init-dagger runs/dagger-arm-modelb/r{1,2,3,4,5}-data \
    > runs/dagger-arm-modelb-lb.log 2>&1 || { echo "ARM STAGE FAILED"; exit 1; }
K=$($PY -B runs/choose_arm_round.py runs/dagger-arm-modelb-lb | tail -1)
ARM=runs/dagger-arm-modelb-lb/r$K/model.pt
echo "chosen arm $ARM"
F="--fresh 0 40 --no-held-out --fresh-file scenarios_obstacles_val_d.json --workers 4"
nice -n 5 $PY -B eval_arm.py --name final-modelb --arm $ARM --base $BASE $F > runs/arm-eval-final-modelb.log 2>&1
nice -n 5 $PY -B eval_arm.py --name final-modelb-hybrid --arm $ARM $F > runs/arm-eval-final-modelb-hybrid.log 2>&1
echo "arm checks finished $(date)"
COMMON="--data runs/rooms-expert --validation-data runs/rooms-expert-val --scenarios scenarios_rooms_train.json scenarios_rooms_val.json --label-expert rrt_visible --rounds 3 --per-round 100 --duration 70 --no-velocity --scans 3"
nice -n 5 timeout 7h $PY -B dagger.py --name rooms-conv --encoder conv --workers 3 $COMMON > runs/dagger-rooms-conv.log 2>&1 &
nice -n 5 timeout 7h $PY -B dagger.py --name rooms-threat --threat --workers 3 $COMMON > runs/dagger-rooms-threat.log 2>&1 &
wait
echo QUEUE 3 FINISHED
