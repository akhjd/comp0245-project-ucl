#!/bin/bash
# Robustness suite on the final base model (29 Oct, after the hard-layout swap test). Same tests as robust-noise-r2.
cd /home/akhjd/ros2_ws/comp0245_project
P=/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python
export MPLCONFIGDIR=runs/.matplotlib
echo "$(date '+%F %T') START robust-hard-r3" >> runs/overnight.log
if nice -n 5 $P -B robustness.py --name hard-r3 --no-expert --workers 4 \
    --models novhard=runs/dagger-novnoise-hard/r3/model.pt > runs/robust-hard-r3.log 2>&1
then echo "$(date '+%F %T') DONE  robust-hard-r3" >> runs/overnight.log
else echo "$(date '+%F %T') FAILED robust-hard-r3 (see runs/robust-hard-r3.log)" >> runs/overnight.log; fi
