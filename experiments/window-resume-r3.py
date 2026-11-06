"""4 Nov: dagger-rooms-window72b was killed (out of memory) while training round 3, after its 100 labelled runs
(r3-data) were saved. Finish that round exactly as dagger.py would: train on the window expert data + r1-r3 DAgger
data with the same settings, score on the 60 room validation scenarios, append round 3 to results.json."""
import json
import sys
from pathlib import Path

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")    # runs/ is a symlink to /data, so not from __file__
sys.path.insert(0, str(ROOT))
from dagger import evaluate
from train_base_torch import train

if __name__ == "__main__":
    root = ROOT/"runs/dagger-rooms-window72b"
    dagger_dirs = [root/f"r{k}-data" for k in (1, 2, 3)]
    rmse = train(["runs/rooms-window72"], dagger_dirs, root/"r3", True, seed=0, lidar_noise=0.0, lidar_dropout=0.0,
                 balance=False, scans=3, hidden=64, encoder="conv", threat=True, mixture=0, modes=False,
                 mode_label="collision", validation="runs/rooms-window72-val")
    scenarios = [s for f in ("scenarios_rooms_train.json", "scenarios_rooms_val.json")
                 for s in json.loads((ROOT/f).read_text())]
    validation = [s for s in scenarios if s["split"] == "validation"]
    totals = evaluate(root/"r3"/"model.pt", root/"r3-validation", validation, 70.0, 4)
    results = json.loads((root/"results.json").read_text())
    results["rounds"].append({"round": 3, "dagger_folders": 3, "offline_rmse_rad_s": rmse, **totals,
                              "note": "trained by runs/window-resume-r3.py after the 09:56 out-of-memory kill"})
    (root/"results.json").write_text(json.dumps(results, indent=1))
    print(f"ROUND 3: offline RMSE {rmse:.3f}, closed loop {totals}", flush=True)
