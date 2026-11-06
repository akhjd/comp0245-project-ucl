"""Validate an earlier round of the combined run on the same 120 validation rooms, exactly as dagger.py validates
the last round (val_all.json, arm folded, 70 s, base only), then score it like run.sh step 5.
Usage: validate_round.py ROUND [WORKERS] [ROOMS.json NAME]   -> runs/dagger-overnight-combined/rROUND-NAME
(default rooms val_all.json, NAME "validation"; fresh_val.json / "fresh" confirms a round picked on val_all)"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")    # runs/ is a symlink to /data, so not from __file__
sys.path.insert(0, str(ROOT))
RUN = ROOT/"runs/dagger-overnight-combined"


def hits(man, group):
    return sum(any(h and m for h, m in zip(man[e["id"]].get("contact_steps_per_obstacle", []),
                                           man[e["id"]].get("obstacle_is_moving", []))) for e in group)


if __name__ == "__main__":
    from dagger import evaluate
    from eval_obstacles import ARM_FOLD
    k, workers = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 12
    rooms_file, name = (Path(sys.argv[3]), sys.argv[4]) if len(sys.argv) > 4 else \
        (ROOT/"runs/overnight-combined/rooms/val_all.json", "validation")
    rooms = [s | {"arm_q0": ARM_FOLD} for s in json.loads(rooms_file.read_text()) if s["split"] == "validation"]
    out = RUN/f"r{k}-{name}"
    print(evaluate(RUN/f"r{k}"/"model.pt", out, rooms, 70.0, workers), flush=True)
    subprocess.run([sys.executable, "-B", "rescore.py", "--as", "rescored-reach.json", str(out)], cwd=ROOT,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    eps = json.loads((out/"rescored-reach.json").read_text())["episodes"]
    man = {e["path"]: e for e in json.loads((out/"manifest.json").read_text())["episodes"]}
    ours = [e for e in eps if e["id"].startswith(("validation", "freshrm"))]
    hers = [e for e in eps if e["id"].startswith(("case", "freshc"))]
    print(f"RESULT r{k} {name} ours: parked {sum(e['parked'] for e in ours)}/{len(ours)}, mover hits {hits(man, ours)}")
    print(f"RESULT r{k} {name} hers: parked {sum(e['parked'] for e in hers)}/{len(hers)}, mover hits {hits(man, hers)}")
