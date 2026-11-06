"""Gate for the crossing DAgger (4 Nov): for each round's model, how well it copies the expert in the 4 s before
window r3's real mover hits (fit-check/crossing-labels, validation rooms, analysis only), plus the round's closed-loop
parked / mover-hit score (arm folded). Usage: gate.py RUN_FOLDER"""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"runs/scans-rooms/fit-check"))
from output_check import crossing_rows, score
from compare_scans import contacts

if __name__ == "__main__":
    run = ROOT/sys.argv[1]
    rows = crossing_rows()
    ones = np.ones(len(rows[1]), bool)
    out = {}
    for k in range(10):
        model = run/f"r{k}"/"model.pt"
        if not model.exists():
            break
        before = score(model, *rows, ones, ones)["near"]
        val = run/f"r{k}-validation" if k else ROOT/"runs/window-fold-r0-start/validation"
        closed = {}
        if (val/"evaluation.json").exists():
            closed = {"parked": json.loads((val/"evaluation.json").read_text())["totals"].get("parked"),
                      "mover_hits": contacts(val)["mover_contact_runs"]}
        out[f"r{k}"] = {"before_hits": before, **closed}
        print(f"r{k}", out[f"r{k}"], flush=True)
    (run/"gate.json").write_text(json.dumps(out, indent=1))
