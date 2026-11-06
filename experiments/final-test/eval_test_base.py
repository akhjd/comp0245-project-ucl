"""The single final test evaluation of the room base model (5 Nov): the 80 test rooms with the arm folded, 70 s, the
same settings as the validation runs (runs/crossing/eval_fold.py). Run once; nothing is chosen from it.
Usage: eval_test_base.py MODEL_PT OUT_FOLDER [WORKERS]"""
import json
import sys
from pathlib import Path

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")    # runs/ is a symlink to /data, so not from __file__
sys.path.insert(0, str(ROOT))
from dagger import evaluate
from eval_obstacles import ARM_FOLD

if __name__ == "__main__":
    model, out = sys.argv[1], ROOT/sys.argv[2]
    if out.exists():
        raise FileExistsError(f"the test evaluation is run once: {out} exists")
    test = [{**s, "arm_q0": ARM_FOLD} for s in json.loads((ROOT/"scenarios_rooms_test.json").read_text())
            if s["split"] == "test"]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    print(len(test), "test rooms", flush=True)
    print(evaluate(ROOT/model, out, test, 70.0, workers), flush=True)
