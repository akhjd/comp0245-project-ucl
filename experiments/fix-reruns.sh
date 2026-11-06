#!/bin/bash
# Reruns after the 29 Oct fixes (Codex review): like-for-like expert reference on every test, and the delay tests
# for the four round-2 models with the corrected delay warm-up. Progress: runs/overnight.log
cd /home/akhjd/ros2_ws/comp0245_project
P=/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python
export MPLCONFIGDIR=runs/.matplotlib
LOG=runs/overnight.log
for job in "fix-expert" "fix-delay --no-expert --tests delay --models vel=runs/dagger-vel2/r2/model.pt nov=runs/dagger-nov/r2/model.pt velnoise=runs/dagger-velnoise/r2/model.pt novnoise=runs/dagger-novnoise/r2/model.pt"; do
    name=${job%% *}
    echo "$(date '+%F %T') START robust-$name" >> $LOG
    if nice -n 5 $P -B robustness.py --name $job --workers 4 > runs/robust-$name.log 2>&1; then
        echo "$(date '+%F %T') DONE  robust-$name" >> $LOG
    else echo "$(date '+%F %T') FAILED robust-$name" >> $LOG; fi
done
