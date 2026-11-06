"""Score weights/base_final.pt on the 60 room validation scenarios, same way as compare_scans.py."""
import json
import sys
from pathlib import Path

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")
sys.path.insert(0, str(ROOT))

from compare_scans import contacts
from dagger import evaluate


if __name__ == "__main__":
    out = ROOT/"runs/scans-rooms/base-final-rerun"
    validation = [s for s in json.loads((ROOT/"scenarios_rooms_val.json").read_text()) if s["split"] == "validation"]
    totals = evaluate(ROOT/"weights/base_final.pt", out/"validation", validation, 70.0, 10)
    result = {**totals, **contacts(out/"validation")}
    (out/"results.json").write_text(json.dumps(result, indent=1))
    print(result, flush=True)
