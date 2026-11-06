#!/bin/bash
# 30 Oct: whole-robot robustness (plan: AGENT_LOG.md 00:05 entry). Fresh hard layouts solved by the expert, then
# robust_full.py (fully learned robot + full-task expert) on val_e and hard3, then efficiency numbers.
cd /home/akhjd/ros2_ws/comp0245_project
P=/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python
export MPLCONFIGDIR=runs/.matplotlib
LOG=runs/overnight.log
W=${WORKERS:-6}
C=scenarios_obstacles_hard3_candidates.json
stage() {   # stage NAME COMMAND... ; stops if a stage fails
    echo "$(date '+%F %T') START $1" >> $LOG
    if nice -n 5 "${@:2}" > runs/$1.log 2>&1; then echo "$(date '+%F %T') DONE  $1" >> $LOG
    else echo "$(date '+%F %T') FAILED $1 (see runs/$1.log)" >> $LOG; exit 1; fi
}
ids() { $P -c "import json,sys; print(' '.join(s['id'] for s in json.load(open('$1'))[$2]))"; }
while pgrep -f "make_obstacle_scenarios.py --out $C" > /dev/null; do sleep 20; done   # started by hand
[ -f $C ] || stage hard3-generate $P -B make_obstacle_scenarios.py --out $C --train 0 --validation 48 --test 0 \
    --seed 606 --prefix hard3 --hard
stage obst-hard3-a $P -B collect_obstacles.py --scenarios $C --out runs/obst-hard3-a --only $(ids $C :16) &
stage obst-hard3-b $P -B collect_obstacles.py --scenarios $C --out runs/obst-hard3-b --only $(ids $C 16:32) &
stage obst-hard3-c $P -B collect_obstacles.py --scenarios $C --out runs/obst-hard3-c --only $(ids $C 32:) &
wait
for f in a b c; do [ -f runs/obst-hard3-$f/manifest.json ] || { echo "$(date '+%F %T') FAILED obst-hard3-$f" >> $LOG; exit 1; }; done
stage hard3-filter $P - <<'EOF'
import json
from pathlib import Path
ok = {e["scenario_id"] for f in "abc" for e in json.loads(Path(f"runs/obst-hard3-{f}/manifest.json").read_text())["episodes"]
      if e["success"]}
candidates = json.loads(Path("scenarios_obstacles_hard3_candidates.json").read_text())
kept = [s for s in candidates if s["id"] in ok][:30]
Path("scenarios_obstacles_hard3.json").write_text(json.dumps(kept, indent=1))
print(f"expert solved {len(ok)}/{len(candidates)}, kept {len(kept)}")
assert len(kept) >= 20, "too few expert-solved hard layouts"
EOF
stage robust-full $P -B robust_full.py --name full --workers $W --validation-file scenarios_obstacles_val_e.json \
    --hard-file scenarios_obstacles_hard3.json
stage efficiency $P -B efficiency.py --run runs/robust-full/fully_learned/clean --out runs/efficiency.json
echo "$(date '+%F %T') full-robot robustness finished" >> $LOG
