"""Rerun window r3 on the 23 validation rooms where it hit a mover, with a shadow WindowExpert labelling every state
(analysis only, 4 Nov). The manifest split is set to "analysis" afterwards so these validation rooms can never be
loaded as training data."""
import json
import sys
from pathlib import Path

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")    # runs/ is a symlink to /data, so not from __file__
sys.path.insert(0, str(ROOT))
from dagger import run_episodes, CONTACT_LIMIT_S, STALL_LIMIT_S

if __name__ == "__main__":
    hits = json.loads((ROOT/"runs/dagger-rooms-window72b/r3-validation-fail-types.json").read_text())
    ids = {h["id"] for h in hits if h["type"].startswith("mover hit")}
    scenarios = [{**s, "expert": "window"} for f in ("scenarios_rooms_train.json", "scenarios_rooms_val.json")
                 for s in json.loads((ROOT/f).read_text()) if s["split"] == "validation" and s["id"] in ids]
    out = ROOT/"runs/scans-rooms/fit-check/crossing-labels"
    run_episodes(ROOT/"runs/dagger-rooms-window72b/r3/model.pt", scenarios, out, 70.0, True, 4,
                 limits=(CONTACT_LIMIT_S, STALL_LIMIT_S))
    manifest = json.loads((out/"manifest.json").read_text())
    for e in manifest["episodes"]:
        e["split"] = "analysis"
    (out/"manifest.json").write_text(json.dumps(manifest, indent=1))
    print(len(manifest["episodes"]), "runs")
