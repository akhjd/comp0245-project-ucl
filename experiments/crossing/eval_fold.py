"""A room model on the 60 room validation scenarios with the arm folded (eval_obstacles.ARM_FOLD), like
runs/arm-tuck/eval_tuck.py, with a duration option (the 100 s run is a report-only time check).
Usage: eval_fold.py MODEL_PT OUT_FOLDER [DURATION_S [WORKERS]]"""
import json
import sys
from pathlib import Path

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")    # runs/ is a symlink to /data, so not from __file__
sys.path.insert(0, str(ROOT))
from dagger import evaluate
from eval_obstacles import ARM_FOLD

if __name__ == "__main__":
    model, out = sys.argv[1], ROOT/sys.argv[2]
    duration = float(sys.argv[3]) if len(sys.argv) > 3 else 70.0
    validation = [{**s, "arm_q0": ARM_FOLD} for f in ("scenarios_rooms_train.json", "scenarios_rooms_val.json")
                  for s in json.loads((ROOT/f).read_text()) if s["split"] == "validation"]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    print(evaluate(ROOT/model, out, validation, duration, workers), flush=True)
