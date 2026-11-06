"""Corner check (5 Nov): the robot starts facing into a room corner (walls on both sides ahead), the target behind it.
Does the driver back out? Runs with the code of the folder given (this project, or Julia's clone).
    window   the window expert (our teacher)          rrt   the privileged RRT expert (Julia's)
    learned  our room base (seed 0 crossing r3 + latched hold, arm folded)
Usage: corner.py CODE_FOLDER DRIVER OUT_FOLDER [GAP_M ...]"""
import contextlib
import json
import os
import sys
from pathlib import Path
import numpy as np

code, driver, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
gaps = [float(g) for g in sys.argv[4:]] or [0.15, 0.08]
sys.path.insert(0, str(code))
from obstacle_env import rollout, ROBOT_RADIUS
from pipeline import read_json

HALF = 3.5
ARM_FOLD = [3.14159, 2.618, 3.14159, 0.524, 0.0, 1.309, 0.0]
results = []
for gap in gaps:
    c = HALF - ROBOT_RADIUS - gap                     # centre this far from both walls ahead (gap = clearance m)
    s = {"id": f"corner_gap{gap:.2f}", "split": "validation", "seed": 0, "start": [c, c, float(np.pi/4)],
         "target": [-1.5, -1.5, 0.58], "park": [-0.85, -0.85], "obstacles": [], "room": [-HALF, -HALF, HALF, HALF],
         "arm_q0": ARM_FOLD}
    config = read_json(code/"settings.json")
    config["duration_s"] = 60.0
    policy = None
    if driver in ("window", "rrt"):
        s["expert"] = driver
    else:
        sys.path.insert(0, "/home/akhjd/ros2_ws/comp0245_project")
        from eval_obstacles import load_model, LearnedBase
        config["lidar_rays"] = 72
        policy = LearnedBase(*load_model("/home/akhjd/ros2_ws/comp0245_project/runs/crossing-hold/r3-hold-latch/model.pt"))
    with open(os.devnull, "w") as quiet, contextlib.redirect_stderr(quiet), contextlib.redirect_stdout(quiet):
        summary = rollout(config, s, out/s["id"], policy, save_every=10)
    z = np.load(out/s["id"]/"trajectory.npz")
    speed = z["observations"][:, 22]                  # forward speed (m/s), robot frame
    dist = np.linalg.norm(z["base_m"][:, :2] - np.array(s["target"][:2]), axis=1)
    r = {"gap_m": gap, "driver": driver, "termination": summary["termination"], "contact_steps": summary["contact_steps"],
         "reversed_s": round(float((speed < -0.02).sum()*0.01), 2), "start_to_target_m": round(float(dist[0]), 2),
         "end_to_target_m": round(float(dist[-1]), 2), "path_m": round(summary["base_path_length_m"], 2),
         "arrived_s": summary["base_arrival_time_s"]}
    results.append(r)
    print(r, flush=True)
(out/"results.json").write_text(json.dumps(results, indent=1))
