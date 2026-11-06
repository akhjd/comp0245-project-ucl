"""Teacher speed test (5 Nov, AGENT_LOG 15:07): FastWindowExpert (turns round, no phantom arm, unfolds before reaching)
at several top speeds on our 60 validation rooms and Julia's 60 single-target validation rooms (case2-4), arm folded,
110 s, the full-task expert (policy None). Per run: our parked score (eval_obstacles.score), contacts, task success.
"old" instead of a speed runs the unchanged WindowExpert (0.25 m/s) with the same arm unfold, as the baseline.
--lidar RAYS RANGE sets the lidar (default 72 rays, 4 m).
Usage: run.py [--tag TAG] [--lidar RAYS RANGE] TOP_SPEED|old [...]   -> runs/teacher-speed/[TAG-]top<speed>/{ours,hers} (or old)"""
import contextlib
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")    # runs/ is a symlink to /data, so not from __file__
sys.path.insert(0, str(ROOT))
OUT = ROOT/"runs/teacher-speed"
FOLD = [3.14159, 2.618, 3.14159, 0.524, 0.0, 1.309, 0.0]


LIDAR = [72, 4.0]


def episode(job):
    scenario, folder, lidar = job
    from eval_obstacles import score
    from obstacle_env import rollout
    from pipeline import read_json
    config = read_json(ROOT/"settings.json")
    config["duration_s"], config["lidar_rays"], config["lidar_range"] = 110.0, int(lidar[0]), float(lidar[1])
    with open(os.devnull, "w") as quiet, contextlib.redirect_stderr(quiet), contextlib.redirect_stdout(quiet):
        summary = rollout(config, scenario, folder, None, save_every=10)
    return summary, score(Path(folder), scenario, summary)


def room_sets():
    ours = [{**s, "arm_q0": FOLD} for f in ("scenarios_rooms_train.json", "scenarios_rooms_val.json")
            for s in json.loads((ROOT/f).read_text()) if s["split"] == "validation"]
    hers = json.loads((ROOT/"runs/julia-check/val_case234_fold.json").read_text())
    return {"ours": ours, "hers": hers}


if __name__ == "__main__":
    from compare_scans import contacts
    args = sys.argv[1:]
    tag = ""
    if args[:1] == ["--tag"]:
        tag, args = args[1] + "-", args[2:]
    if args[:1] == ["--lidar"]:
        LIDAR, args = [int(args[1]), float(args[2])], args[3:]
    for arg in args:
        top = 0.25 if arg == "old" else float(arg)
        label = tag + ("old" if arg == "old" else f"top{top:g}")
        expert = "window" if arg == "old" else "window_fast"
        jobs, where = [], []
        for name, rooms in room_sets().items():
            d = OUT/label/name
            d.mkdir(parents=True)
            for s in rooms:
                jobs.append(({**s, "expert": expert, "top_speed": top, "expert_unfold": True}, str(d/s["id"]), LIDAR))
                where.append(name)
        with get_context("spawn").Pool(12) as pool:
            results = pool.map(episode, jobs, chunksize=1)
        for name in ("ours", "hers"):
            d = OUT/label/name
            rs = [(j, r) for j, r, w in zip(jobs, results, where) if w == name]
            (d/"manifest.json").write_text(json.dumps({"saved_every": 10, "lidar_rays": LIDAR[0], "lidar_range": LIDAR[1], "episodes": [
                {"path": j[0]["id"], **summary} for j, (summary, _) in rs]}, indent=1))
            scores = [sc for _, (_, sc) in rs]
            arrived = [summary["base_arrival_time_s"] for _, (summary, _) in rs if summary["base_arrival_time_s"] is not None]
            totals = {"run": label, "lidar": LIDAR, "expert": expert, "top_speed": top, "rooms": len(rs), "parked": sum(s["parked"] for s in scores),
                      "task_success": sum(summary["success"] for _, (summary, _) in rs),
                      "median_arrival_s": sorted(arrived)[len(arrived)//2] if arrived else None, **contacts(d)}
            (d/"evaluation.json").write_text(json.dumps({"totals": totals, "episodes": scores}, indent=1))
            print(f"TOTALS {label} {name}: {totals}", flush=True)
