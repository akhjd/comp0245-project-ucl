"""Window r3 on the 60 room validation scenarios with the arm folded inside the base footprint while driving
(scenario "arm_q0", held by LearnedBase as before). Compare with runs/dagger-rooms-window72b/r3-validation."""
import json
import sys
from pathlib import Path

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")    # runs/ is a symlink to /data, so not from __file__
sys.path.insert(0, str(ROOT))
from dagger import evaluate

TUCK = [3.14159, 2.618, 3.14159, 0.524, 0.0, 1.309, 0.0]   # search.py: inside x +-0.342, y +-0.313, 9 cm from the base

if __name__ == "__main__":
    model, out = sys.argv[1], ROOT/sys.argv[2]
    validation = [{**s, "arm_q0": TUCK} for f in ("scenarios_rooms_train.json", "scenarios_rooms_val.json")
                  for s in json.loads((ROOT/f).read_text()) if s["split"] == "validation"]
    print(evaluate(ROOT/model, out, validation, 70.0, 4), flush=True)
