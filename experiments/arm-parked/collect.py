"""Expert arm data from parked spots (5 Nov): the base stands still where make_scenarios.py put it (robot.arrived, as in
measure_reach.py) and the expert arm controller reaches from the normal start pose. Saved like an expert dataset
(one folder per spot with trajectory.npz, plus manifest.json with base_arrival_time_s = 0), every 10th control step,
so train_arm.py reads it as it is. Ends on the course rule (hand within 1 cm for 0.25 s) or after DURATION.
Usage: collect.py SCENARIOS_JSON OUT_FOLDER [WORKERS]"""
import contextlib
import io
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")    # runs/ is a symlink to /data, so not from __file__
sys.path.insert(0, str(ROOT))
DURATION = 12.0      # s the arm gets (measure_reach.py)
SAVE_EVERY = 10      # control steps per saved row


def attempt(job):
    scenario, folder = job
    from obstacle_env import ObstacleRobot
    from models import clip_action
    from pipeline import read_json
    config = read_json(ROOT/"settings.json")
    with contextlib.redirect_stdout(io.StringIO()):
        robot = ObstacleRobot(config, scenario)
    try:
        robot.arrived = True                  # the base is parked here: only the arm moves
        keys = ["observations", "actions", "time_s", "ee_m", "error_m", "base_m", "arm_q_rad"]
        rows = {k: [] for k in keys}
        state, t, dwell, best, step = robot.measure(), 0.0, 0.0, np.inf, 0
        while t < DURATION:
            action = clip_action(robot.expert_action(state, t), config)
            if step % SAVE_EVERY == 0:
                for k, v in zip(keys, (state["x"], action, t, state["ee"], state["error"], state["base"], state["q"][4:11])):
                    rows[k].append(np.copy(v))
            robot.step(action)
            t += robot.control_dt
            step += 1
            state = robot.measure()
            best = min(best, state["error"])
            dwell = dwell + robot.control_dt if state["error"] <= config["success_tolerance_m"] else 0.0
            if dwell >= config["success_hold_s"]:
                break
        contact = bool(robot.contacts().any()) if len(robot.obstacle_ids) else False
        Path(folder).mkdir(parents=True)
        np.savez_compressed(Path(folder)/"trajectory.npz", **{k: np.array(v) for k, v in rows.items()})
        success = bool(dwell >= config["success_hold_s"]) and not contact
        summary = {"scenario_id": scenario["id"], "success": success, "reached_target": success,
                   "simulation_time_s": round(t, 3), "minimum_error_m": float(best), "final_error_m": float(state["error"]),
                   "base_arrival_time_s": 0.0, "base_moved_m": float(np.linalg.norm(state["base"][:2])),
                   "target_z_m": scenario["target"][2], "saved_every": SAVE_EVERY}
        (Path(folder)/"summary.json").write_text(json.dumps(summary, indent=1))
        (Path(folder)/"episode.json").write_text(json.dumps({"scenario": scenario}, indent=1))
        return summary
    finally:
        robot.close()


if __name__ == "__main__":
    scenarios = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    if out.exists():
        raise FileExistsError(f"Choose a new output folder: {out}")
    out.mkdir(parents=True)
    episodes = []
    with Pool(workers) as pool:
        for s, summary in zip(scenarios, pool.imap(attempt, [(s, out/s["id"]) for s in scenarios])):
            episodes.append({"path": s["id"], **summary})
            if len(episodes) % 50 == 0:
                print(f"{len(episodes)}/{len(scenarios)}: {sum(e['success'] for e in episodes)} successful", flush=True)
    (out/"manifest.json").write_text(json.dumps({"saved_every": SAVE_EVERY, "scenarios_file": str(sys.argv[1]),
                                                 "episodes": episodes}, indent=1))
    print(f"{sum(e['success'] for e in episodes)}/{len(episodes)} successful")
