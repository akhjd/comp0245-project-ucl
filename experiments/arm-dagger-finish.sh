#!/bin/bash
# 29 Oct: arm DAgger rounds 4-5 (resume), choose the round by the rule fixed in AGENT_LOG.md (16:45), then the
# fully learned robot on validationc_060-099 (never used to choose anything), with two comparisons.
cd /home/akhjd/ros2_ws/comp0245_project
P=/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python
export MPLCONFIGDIR=runs/.matplotlib
LOG=runs/overnight.log
stage() {   # stage NAME COMMAND... ; stops if a stage fails
    echo "$(date '+%F %T') START $1" >> $LOG
    if nice -n 5 "${@:2}" > runs/$1.log 2>&1; then echo "$(date '+%F %T') DONE  $1" >> $LOG
    else echo "$(date '+%F %T') FAILED $1 (see runs/$1.log)" >> $LOG; exit 1; fi
}
stage dagger-arm-a-r45 $P -B dagger_arm.py --name a --rounds 5 --resume --workers 4
stage arm-choose $P - <<'EOF'
import json
from pathlib import Path
rounds = json.loads(Path("runs/dagger-arm-a/results.json").read_text())["rounds"]
best = max(rounds, key=lambda r: (r["success"], r["fresh-normal"], -r["round"]))
Path("runs/dagger-arm-a/chosen.json").write_text(json.dumps({"round": best["round"], "rule":
    "most successes on the 90 eval_arm scenarios; tie -> fresh-normal; then the earlier round", "scores": best}, indent=1))
for r in rounds:
    print(f"r{r['round']}: {r['success']}/{r['episodes']}  (height {r['obst-v4-height']}, armstart {r['obst-v4-armstart']}, "
          f"both {r['obst-v4-both']}, fresh {r['fresh-normal']})")
print("chosen: r%d" % best["round"])
EOF
ARM=runs/dagger-arm-a/r$($P -c "import json; print(json.load(open('runs/dagger-arm-a/chosen.json'))['round'])")/model.pt
BASE=runs/dagger-novnoise-hard/r3/model.pt
F="--fresh 60 100 --no-held-out --workers 4"
stage arm-eval-full        $P -B eval_arm.py --name full        --arm $ARM --base $BASE $F
stage arm-eval-full-hybrid $P -B eval_arm.py --name full-hybrid --arm $ARM $F
stage arm-eval-full-expert $P -B eval_arm.py --name full-expert --arm expert $F
echo "$(date '+%F %T') arm DAgger + full-robot test finished" >> $LOG
