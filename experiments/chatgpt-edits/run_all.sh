#!/bin/bash
# Evaluate ChatGPT's edited models one after another (60 validation rooms, arm folded, 70 s), then rescore.
ROOT=/home/akhjd/ros2_ws/comp0245_project
P=/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python
cd $ROOT
for n in "$@"; do
    $P -B runs/crossing/eval_fold.py runs/chatgpt-edits/$n/model.pt runs/chatgpt-edits/$n-validation 70 8 > runs/chatgpt-edits/$n-validation.log 2>&1
    $P -B rescore.py runs/chatgpt-edits/$n-validation --as rescored-reach.json 2>&1 | grep parked
done
echo "ALL EDITS DONE $(date)"
