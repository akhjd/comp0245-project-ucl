"""Arm-only test (5 Nov): the base placed exactly where the room base parked in the validation full-robot runs that
handed over to the arm (runs/arm-eval-rooms-full-val, pose at the moment it passed the parking rule), wheels at 0,
room walls and still obstacles kept, movers removed. The arm starts folded, unfolds (eval_arm.LearnedArm) and the
learned arm reaches. Success = the course rule (hand within 1 cm for 0.25 s) with no contact.
With SPOTS_JSON (make_scenarios.py output), those synthetic parked spots are used instead, also folded -> unfold.
Usage: arm_test.py ARM_MODEL_PT OUT_FOLDER [WORKERS [SPOTS_JSON]]"""
import contextlib
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path
import numpy as np

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")    # runs/ is a symlink to /data, so not from __file__
sys.path.insert(0, str(ROOT))
SOURCE = ROOT/"runs/arm-eval-rooms-full-val"
DURATION = 25.0      # s: unfolding takes ~7 s, the arm then reached in a median 3 s


class Parked:
    """Stands in for the base: already parked, wheels at 0."""
    arrived = True

    def act(self, state):
        return np.zeros(11)


def spots():
    """(scenario, parked pose) for every validation run whose base handed over to the arm."""
    from eval_obstacles import ARM_FOLD
    out = []
    for e in json.loads((SOURCE/"results.json").read_text())["episodes"]:
        if e["parked_at_s"] is None or e["unfold_s"] is None:
            continue
        folder = SOURCE/e["set"]/e["id"]
        z = np.load(folder/"trajectory.npz")
        i = min(int(np.searchsorted(z["time_s"], e["parked_at_s"])), len(z["time_s"]) - 1)
        obs = z["observations"][i]
        pose = [float(obs[0]), float(obs[1]), float(np.arctan2(obs[3], obs[4]))]
        s = json.loads((folder/"episode.json").read_text())["scenario"]
        still = [o for o in s["obstacles"] if not np.any(o.get("vel", [0, 0]))]
        out.append({**s, "start": pose, "obstacles": still, "arm_q0": ARM_FOLD})
    return out


def episode(job):
    arm_path, scenario, folder = job
    import torch
    torch.set_num_threads(1)
    from obstacle_env import rollout
    from pipeline import read_json
    from eval_arm import LearnedArm, load_arm
    config = read_json(ROOT/"settings.json")
    config["duration_s"] = DURATION
    policy = LearnedArm(*load_arm(arm_path), scenario["target"], Parked(), unfold=True)
    with open(os.devnull, "w") as quiet, contextlib.redirect_stderr(quiet):
        summary = rollout(config, scenario, folder, policy, save_every=10)
    return {"id": scenario["id"], "reached": summary["reached_target"], "contact_steps": summary["contact_steps"],
            "success": summary["reached_target"] and summary["contact_steps"] == 0,
            "minimum_error_m": summary["minimum_error_m"], "final_error_m": summary["final_error_m"],
            "unfold_s": policy.reach_from, "target_z_m": scenario["target"][2],
            "reach_time_s": (summary["simulation_time_s"] - policy.reach_from) if policy.reach_from is not None else None}


if __name__ == "__main__":
    arm, out = sys.argv[1], ROOT/sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    if out.exists():
        raise FileExistsError(f"Choose a new output folder: {out}")
    out.mkdir(parents=True)
    if len(sys.argv) > 4:
        from eval_obstacles import ARM_FOLD
        chosen = [{**s, "arm_q0": ARM_FOLD} for s in json.loads(Path(sys.argv[4]).read_text())]
    else:
        chosen = spots()
    with get_context("spawn").Pool(workers) as pool:
        results = pool.map(episode, [(arm, s, str(out/s["id"])) for s in chosen], chunksize=1)
    n, ok = len(results), sum(r["success"] for r in results)
    (out/"results.json").write_text(json.dumps({"arm": arm, "source": sys.argv[4] if len(sys.argv) > 4 else str(SOURCE), "duration_s": DURATION,
        "spots": n, "success": ok, "reached": sum(r["reached"] for r in results),
        "contact": sum(r["contact_steps"] > 0 for r in results), "episodes": results}, indent=1))
    print(f"{arm}: success {ok}/{n}, reached {sum(r['reached'] for r in results)}, "
          f"contact {sum(r['contact_steps'] > 0 for r in results)}", flush=True)
