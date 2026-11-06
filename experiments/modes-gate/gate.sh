#!/bin/bash
# 3 Nov (side session lidar72-modes): offline gate for the evade/target switch (AGENT_LOG 22:26 PLAN).
# Same data and settings for both: conv ray filter, 3 scans, no velocity, 36 rays, 10 epochs, seed 0.
cd /home/akhjd/ros2_ws/comp0245_project
P=/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python
COMMON="--data runs/rooms-expert --validation runs/rooms-expert-val --encoder conv --scans 3 --no-velocity --epochs 10 --seed 0"
nice -n 5 $P -B train_base_torch.py $COMMON --modes --mode-label collision --out runs/modes-gate/modes > runs/modes-gate/modes.log 2>&1 && \
nice -n 5 $P -B train_base_torch.py $COMMON --out runs/modes-gate/plain-conv > runs/modes-gate/plain-conv.log 2>&1
echo "gate training exit $?"
