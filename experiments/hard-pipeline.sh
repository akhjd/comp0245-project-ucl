#!/bin/bash
# Hard-layout DAgger, agreed with Akhilesh on 29 Oct 2026 (plan and swap rule in AGENT_LOG.md, 11:05 entry).
# 1 expert drives on the new hard layout candidates -> 2 keep only expert-solved layouts -> 3 noise DAgger with
# half normal / half hard drives and a balanced loss -> 4 choose the round -> 5 swap test against novnoise r2.
# Progress: runs/overnight.log ; each stage also writes runs/<stage>.log ; live view: bash status.sh
cd /home/akhjd/ros2_ws/comp0245_project
P=/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python
export MPLCONFIGDIR=runs/.matplotlib
LOG=runs/overnight.log
DATA="runs/obst-v2b runs/obst-v3-extra-0 runs/obst-v3-extra-1 runs/obst-v3-extra-2 runs/obst-v3-extra-3"
FINAL=runs/dagger-novnoise/r2/model.pt
stage() {   # stage NAME COMMAND... ; stops the pipeline if a stage fails
    echo "$(date '+%F %T') START $1" >> $LOG
    if nice -n 5 "${@:2}" > runs/$1.log 2>&1; then echo "$(date '+%F %T') DONE  $1" >> $LOG
    else echo "$(date '+%F %T') FAILED $1 (see runs/$1.log); hard pipeline stopped" >> $LOG; exit 1; fi
}
echo "$(date '+%F %T') hard pipeline started" >> $LOG

# 1. Expert drives on all candidates, 4 simulators at once (the 140 training layouts in two halves)
ids() { $P -c "import json; print(' '.join(s['id'] for s in json.load(open('$1'))[$2]))"; }
stage obst-hard2-train-a  $P -B collect_obstacles.py --scenarios scenarios_obstacles_hard2_candidates_train.json \
    --out runs/obst-hard2-train-a --only $(ids scenarios_obstacles_hard2_candidates_train.json :70) &
stage obst-hard2-train-b  $P -B collect_obstacles.py --scenarios scenarios_obstacles_hard2_candidates_train.json \
    --out runs/obst-hard2-train-b --only $(ids scenarios_obstacles_hard2_candidates_train.json 70:) &
stage obst-hard2-select   $P -B collect_obstacles.py --scenarios scenarios_obstacles_hard2_candidates_select.json \
    --out runs/obst-hard2-select &
stage obst-hard2-gate     $P -B collect_obstacles.py --scenarios scenarios_obstacles_hard2_candidates_gate.json \
    --out runs/obst-hard2-gate &
wait
for f in train-a train-b select gate; do grep -q "DONE  obst-hard2-$f" $LOG || { echo "$(date '+%F %T') hard pipeline stopped: collection obst-hard2-$f failed" >> $LOG; exit 1; }; done

# 2. Keep only the layouts the expert solved (success = target reached, no contact); the gate set keeps its first 30
stage hard2-filter $P - <<'EOF'
import json
from pathlib import Path
def solved(*folders):
    return {e["scenario_id"] for f in folders for e in json.loads(Path(f, "manifest.json").read_text())["episodes"] if e["success"]}
for name, folders, keep in [("train", ["runs/obst-hard2-train-a", "runs/obst-hard2-train-b"], None),
                            ("select", ["runs/obst-hard2-select"], None), ("gate", ["runs/obst-hard2-gate"], 30)]:
    candidates = json.loads(Path(f"scenarios_obstacles_hard2_candidates_{name}.json").read_text())
    ok = solved(*folders)
    kept = [s for s in candidates if s["id"] in ok][:keep]
    Path(f"scenarios_obstacles_hard2_{name}.json").write_text(json.dumps(kept, indent=1))
    print(f"{name}: expert solved {len(ok)}/{len(candidates)}, kept {len(kept)}")
EOF

# 3. Noise DAgger with hard layouts mixed in (same settings as dagger-novnoise otherwise)
stage dagger-novnoise-hard $P -B dagger.py --name novnoise-hard --no-velocity --rounds 3 --workers 4 \
    --lidar-noise 0.2 --lidar-dropout 0.2 --balance \
    --hard-scenarios scenarios_obstacles_hard2_train.json --select-scenarios scenarios_obstacles_hard2_select.json \
    --data $DATA runs/obst-hard2-train-a runs/obst-hard2-train-b

# 4. Choose the round (rule fixed beforehand): max reference loose + hard-select loose, tie -> reference strict
stage hard2-choose $P - <<'EOF'
import json
from pathlib import Path
rounds = json.loads(Path("runs/dagger-novnoise-hard/results.json").read_text())["rounds"]
best = max(rounds, key=lambda r: (r["loose"] + r["select_loose"], r["strict"]))
Path("runs/dagger-novnoise-hard/chosen.json").write_text(json.dumps({"round": best["round"], "rule":
    "max(reference loose + hard-select loose), tie -> reference strict", "scores": best}, indent=1))
for r in rounds:
    print(f"r{r['round']}: reference {r['strict']}/{r['loose']} of {r['episodes']}, hard-select "
          f"{r['select_strict']}/{r['select_loose']} of {r['select_episodes']}")
print("chosen: r%d" % best["round"])
EOF
NEW=runs/dagger-novnoise-hard/r$($P -c "import json; print(json.load(open('runs/dagger-novnoise-hard/chosen.json'))['round'])")/model.pt

# 5. Swap test: both models under identical conditions
M="--no-expert --workers 4 --models final=$FINAL new=$NEW"
stage robust-gate-normal  $P -B robustness.py --name gate-normal  $M --tests clean --start 30 --n 70
stage robust-gate-oldhard $P -B robustness.py --name gate-oldhard $M --tests hard --duration 60
stage robust-gate-newhard $P -B robustness.py --name gate-newhard $M --tests hard --duration 60 \
    --hard-file scenarios_obstacles_hard2_gate.json
stage hard2-verdict $P - <<'EOF'
import json
from pathlib import Path
def row(run, label):
    return next(t for t in json.loads(Path(f"runs/robust-{run}/results.json").read_text())["table"] if t["controller"] == label)
n_f, n_n = row("gate-normal", "final"), row("gate-normal", "new")
gain = sum(row(r, "new")["loose"] - row(r, "final")["loose"] for r in ("gate-oldhard", "gate-newhard"))
normal_ok = n_n["loose"] >= n_f["loose"] - 2 and n_n["contact"] <= n_f["contact"] + 1
hard_ok = gain >= 5
for r in ("gate-normal", "gate-oldhard", "gate-newhard"):
    f, n = row(r, "final"), row(r, "new")
    print(f"{r:13s} final {f['strict']}/{f['loose']}/{f['contact']}  new {n['strict']}/{n['loose']}/{n['contact']}  (strict/loose/contact of {f['episodes']})")
verdict = "SWAP: the new model becomes final" if normal_ok and hard_ok else "KEEP novnoise r2"
print(f"normal not worse: {normal_ok}; hard loose gain {gain} (need >= 5): {hard_ok} -> {verdict}")
Path("runs/robust-gate-normal/verdict.json").write_text(json.dumps({"normal_ok": normal_ok, "hard_gain": gain,
                                                                   "hard_ok": hard_ok, "verdict": verdict}, indent=1))
EOF
echo "$(date '+%F %T') hard pipeline finished: $(tail -1 runs/hard2-verdict.log)" >> $LOG
