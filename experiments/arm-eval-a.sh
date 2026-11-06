#!/bin/bash
# Closed-loop test of arm model A (expert wheels + learned arm), then the full-task expert on the same scenarios.
# Starts once the robustness suite on the final base model has finished (one batch of simulators at a time).
cd /home/akhjd/ros2_ws/comp0245_project
P=/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python
export MPLCONFIGDIR=runs/.matplotlib
LOG=runs/overnight.log
while pgrep -f "robustness.py --name hard-r3" > /dev/null; do sleep 30; done
for job in "arm-a --arm runs/arm-a/model.pt" "expert --arm expert"; do
    name=${job%% *}
    echo "$(date '+%F %T') START arm-eval-$name" >> $LOG
    if nice -n 5 $P -B eval_arm.py --name $job --workers 4 > runs/arm-eval-$name.log 2>&1; then
        echo "$(date '+%F %T') DONE  arm-eval-$name" >> $LOG
    else echo "$(date '+%F %T') FAILED arm-eval-$name (see runs/arm-eval-$name.log)" >> $LOG; fi
done
