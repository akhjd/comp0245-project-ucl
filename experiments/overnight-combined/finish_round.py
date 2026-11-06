"""Finish a round validation that was cut off (6 Nov: the PC ran out of memory with 113 of 120 rooms done): runs only
the rooms without a summary.json (unfinished folders moved to the scratchpad first), then writes manifest.json and
evaluation.json for all rooms exactly as dagger.run_episodes / evaluate do, and scores them like validate_round.py.
Usage: finish_round.py ROUND [WORKERS] [ROOMS.json NAME]"""
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")    # runs/ is a symlink to /data, so not from __file__
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"runs/overnight-combined"))
RUN = ROOT/"runs/dagger-overnight-combined"
ASIDE = Path("/tmp/claude-1000/-home-akhjd-ros2-ws-comp0245-project/6b3b2ff6-48e0-4813-9e6c-504ba49cfa02/scratchpad/unfinished")

if __name__ == "__main__":
    import numpy as np
    import torch
    from dagger import run_episodes, SAVE_EVERY
    from eval_obstacles import ARM_FOLD, score
    import validate_round
    k, workers = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 6
    rooms_file, name = (Path(sys.argv[3]), sys.argv[4]) if len(sys.argv) > 4 else \
        (ROOT/"runs/overnight-combined/rooms/val_all.json", "validation")
    rooms = [s | {"arm_q0": ARM_FOLD} for s in json.loads(rooms_file.read_text()) if s["split"] == "validation"]
    out, model = RUN/f"r{k}-{name}", RUN/f"r{k}"/"model.pt"
    missing = [s for s in rooms if not (out/s["id"]/"summary.json").exists()]
    stamp = time.strftime("%H%M%S")
    for s in missing:
        if (out/s["id"]).exists():
            shutil.move(str(out/s["id"]), str(ASIDE/f"r{k}-{name}-{stamp}"/s["id"]))
    print(len(rooms) - len(missing), "done,", len(missing), "to run:", [s["id"] for s in missing], flush=True)
    if missing:
        rest = out.with_name(out.name + f"-rest-{stamp}")
        run_episodes(model, missing, rest, 70.0, False, workers)
        for s in missing:
            shutil.move(str(rest/s["id"]), str(out/s["id"]))
        shutil.rmtree(rest)      # only its manifest is left; the full one is written below
    summaries = [json.loads((out/s["id"]/"summary.json").read_text()) for s in rooms]
    scores = [score(out/s["id"], s, m) for s, m in zip(rooms, summaries)]
    saved = torch.load(model, map_location="cpu", weights_only=False)
    manifest = {"saved_every": SAVE_EVERY, "lidar_rays": saved.get("rays", 36), "lidar_range": saved.get("lidar_range", 4.0),
                "episodes": [{"path": s["id"], "scenario_id": s["id"], "split": s["split"], **m} for s, m in zip(rooms, summaries)]}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=1))
    totals = {"episodes": len(scores), "parked": sum(r.get("parked", False) for r in scores),
              "strict": sum(r["strict"] for r in scores), "loose": sum(r["loose"] for r in scores),
              "no_contact": sum(r["contact_steps"] == 0 for r in scores), "reachable": sum(r["reachable"] for r in scores),
              "median_distance_error_m": float(np.median([r["distance_error_m"] for r in scores]))}
    (out/"evaluation.json").write_text(json.dumps({"model": str(model), "arm": "fixed PD hold (not learned)",
                                                   "totals": totals, "episodes": scores,
                                                   "note": f"finished by finish_round.py ({len(missing)} rooms rerun)"}, indent=1))
    print(totals, flush=True)
    import subprocess
    subprocess.run([sys.executable, "-B", "rescore.py", "--as", "rescored-reach.json", str(out)], cwd=ROOT,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    eps = json.loads((out/"rescored-reach.json").read_text())["episodes"]
    man = {e["path"]: e for e in manifest["episodes"]}
    ours = [e for e in eps if e["id"].startswith(("validation", "freshrm"))]
    hers = [e for e in eps if e["id"].startswith(("case", "freshc"))]
    print(f"RESULT r{k} {name} ours: parked {sum(e['parked'] for e in ours)}/{len(ours)}, mover hits {validate_round.hits(man, ours)}")
    print(f"RESULT r{k} {name} hers: parked {sum(e['parked'] for e in hers)}/{len(hers)}, mover hits {validate_round.hits(man, hers)}")
