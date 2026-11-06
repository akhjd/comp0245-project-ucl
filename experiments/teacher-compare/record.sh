#!/bin/bash
# Teacher comparison (3-4 Nov): record the same 40 train + 20 validation rooms with one teacher, 36 rays, 110 s.
# Usage: bash runs/teacher-compare/record.sh TEACHER WORKERS   (TEACHER: rrt_best | window)
cd /home/akhjd/ros2_ws/comp0245_project
PY=/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python
T=$1; W=${2:-4}
TR=$($PY -c "import json;print(' '.join(json.load(open('runs/teacher-compare/ids.json'))['train']))")
VA=$($PY -c "import json;print(' '.join(json.load(open('runs/teacher-compare/ids.json'))['validation']))")
nice -n 5 $PY -B collect_obstacles.py --scenarios scenarios_rooms_train.json --out runs/teacher-compare/$T --teacher $T --duration 110 --workers $W --only $TR > runs/teacher-compare/$T.log 2>&1
nice -n 5 $PY -B collect_obstacles.py --scenarios scenarios_rooms_val.json --out runs/teacher-compare/$T-val --teacher $T --duration 110 --workers $W --only $VA > runs/teacher-compare/$T-val.log 2>&1
echo "$T finished $(date)"
