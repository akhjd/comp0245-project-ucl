#!/bin/bash
# RESUME (after the 29 Oct app restart) of the hard-layout DAgger, agreed with Akhilesh on 29 Oct 2026 (plan and swap rule in AGENT_LOG.md, 11:05 entry).
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
echo "$(date '+%F %T') hard pipeline resumed at robust-gate-newhard" >> $LOG
NEW=runs/dagger-novnoise-hard/r$($P -c "import json; print(json.load(open('runs/dagger-novnoise-hard/chosen.json'))['round'])")/model.pt

# 5. Swap test: both models under identical conditions
M="--no-expert --workers 4 --models final=$FINAL new=$NEW"
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
