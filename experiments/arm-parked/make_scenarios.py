"""Parked spots for new arm data (5 Nov): the base already parked where the room base parks, the arm reaching from its
normal start pose (where it is after unfolding). Target height U(0.36, 0.80) m (the room targets), bearing U(-15, 15)
deg (the room base parks within 10 deg), distance U(0.80, 1.00) x reach_limit(height, bearing). No obstacles.
Usage: make_scenarios.py OUT_JSON [COUNT [SEED]]"""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")    # runs/ is a symlink to /data, so not from __file__
sys.path.insert(0, str(ROOT))
from eval_obstacles import reach_limit

if __name__ == "__main__":
    out = Path(sys.argv[1])
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 20261105)
    scenarios = []
    for k in range(count):
        z = rng.uniform(0.36, 0.80)
        bearing = np.radians(rng.uniform(-15, 15))
        d = rng.uniform(0.80, 1.00)*reach_limit(z, bearing)
        scenarios.append({"id": f"parked_{k:04d}", "split": "train", "seed": int(rng.integers(1_000_000)),
                          "start": [0.0, 0.0, 0.0], "target": [float(d*np.cos(bearing)), float(d*np.sin(bearing)), float(z)],
                          "park": [0.0, 0.0], "obstacles": []})
    out.write_text(json.dumps(scenarios, indent=1))
    print(f"wrote {count} parked spots to {out}")
