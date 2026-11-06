"""Baseline for the combined overnight run: the current room model (runs/crossing-hold/r3-hold-latch, its own 72-ray
4 m lidar) base-only on Julia's 60 single-target validation rooms, arm folded, 70 s (dagger.evaluate).
Usage: baseline.py OUT_FOLDER WORKERS [LIMIT]"""
import json
import sys
from pathlib import Path

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")    # runs/ is a symlink to /data, so not from __file__
sys.path.insert(0, str(ROOT))
FOLD = [3.14159, 2.618, 3.14159, 0.524, 0.0, 1.309, 0.0]

if __name__ == "__main__":
    from dagger import evaluate
    hers = [{k: v for k, v in s.items() if k not in ("expert", "top_speed", "expert_unfold")} | {"arm_q0": FOLD}
            for s in json.loads((ROOT/"runs/julia-check/val_case234_fold.json").read_text())]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    print(evaluate(ROOT/"runs/crossing-hold/r3-hold-latch/model.pt", ROOT/sys.argv[1], hers[:limit] if limit else hers,
                   70.0, int(sys.argv[2])), flush=True)
