#!/bin/bash
# 29 Oct: arm DAgger with the learned base parking (option A), then the final full-robot check on a NEW fresh set
# (scenarios_obstacles_val_d.json). Plan and round-choice rule: AGENT_LOG.md 17:45 entry.
cd /home/akhjd/ros2_ws/comp0245_project
P=/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python
export MPLCONFIGDIR=runs/.matplotlib
LOG=runs/overnight.log
W=${WORKERS:-6}
BASE=runs/dagger-novnoise-hard/r3/model.pt
stage() {   # stage NAME COMMAND... ; stops if a stage fails
    echo "$(date '+%F %T') START $1" >> $LOG
    if nice -n 5 "${@:2}" > runs/$1.log 2>&1; then echo "$(date '+%F %T') DONE  $1" >> $LOG
    else echo "$(date '+%F %T') FAILED $1 (see runs/$1.log)" >> $LOG; exit 1; fi
}
stage dagger-arm-b $P -B dagger_arm.py --name b --rounds 3 --workers $W --base $BASE \
    --init-model runs/dagger-arm-a/r5/model.pt --init-dagger runs/dagger-arm-a/r1-data runs/dagger-arm-a/r2-data \
    runs/dagger-arm-a/r3-data runs/dagger-arm-a/r4-data runs/dagger-arm-a/r5-data
stage arm-b-choose $P - <<'EOF'
import json
from pathlib import Path
rounds = json.loads(Path("runs/dagger-arm-b/results.json").read_text())["rounds"]
best = max(rounds, key=lambda r: (r["full_success"], r["success"], -r["round"]))
Path("runs/dagger-arm-b/chosen.json").write_text(json.dumps({"round": best["round"], "rule":
    "most fully-learned successes on the 90; tie -> expert-parking successes; then the earlier round",
    "scores": best}, indent=1))
for r in rounds:
    print(f"r{r['round']}: fully learned {r['full_success']}/{r['episodes']}, expert parking {r['success']}/{r['episodes']}")
print("chosen: r%d" % best["round"])
EOF
ARM=runs/dagger-arm-b/r$($P -c "import json; print(json.load(open('runs/dagger-arm-b/chosen.json'))['round'])")/model.pt
F="--fresh 0 40 --no-held-out --fresh-file scenarios_obstacles_val_d.json --workers $W"
stage arm-eval-final        $P -B eval_arm.py --name final        --arm $ARM --base $BASE $F
stage arm-eval-final-hybrid $P -B eval_arm.py --name final-hybrid --arm $ARM $F
stage arm-eval-final-expert $P -B eval_arm.py --name final-expert --arm expert $F
echo "$(date '+%F %T') arm DAgger b + final check finished" >> $LOG
