"""Measure where the arm can really reach from: the base stands still, a target is placed at a set horizontal
distance from the base centre, bearing (0 = straight ahead) and height, and the expert arm controller (the one
that made the arm's training data) tries to reach it. Success = the course rule (hand within 1 cm for 0.25 s).

This defines the "reachable area" of the base parking score (eval_obstacles.score "parked"), measured instead of
taken from the arm's full length (reach_zone.ARM_REACH, the straight arm). No obstacles, no walls, no movers.

Example:
    python measure_reach.py --out runs/reach-area --workers 2
    python measure_reach.py --out runs/reach-area-fine --distances 0.90 1.10 0.02     # finer, near the edge
"""
import argparse
import contextlib
import io
import itertools
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DISTANCES = np.round(np.arange(0.3, 1.25, 0.1), 2)     # m from the base centre, horizontal
BEARINGS = (0.0, 30.0)                                  # degrees: straight ahead and the facing limit
HEIGHTS = (0.36, 0.58, 0.80)                            # m: lowest, median and highest room-scenario targets
DURATION = 12.0                                         # s the arm gets


def attempt(job):
    d, bearing, z = job
    from obstacle_env import ObstacleRobot
    from pipeline import read_json
    config = read_json(ROOT/"settings.json")
    b = np.radians(bearing)
    target = [float(d*np.cos(b)), float(d*np.sin(b)), z]
    scenario = {"id": f"reach_{d}_{bearing}_{z}", "split": "validation", "seed": 0, "start": [0.0, 0.0, 0.0],
                "target": target, "park": [0.0, 0.0], "obstacles": []}
    with contextlib.redirect_stdout(io.StringIO()):
        robot = ObstacleRobot(config, scenario)
    try:
        robot.arrived = True                  # the base is parked here: only the arm moves
        state, t, dwell, best = robot.measure(), 0.0, 0.0, np.inf
        while t < DURATION:
            robot.step(robot.expert_action(state, t))
            t += robot.control_dt
            state = robot.measure()
            best = min(best, state["error"])
            dwell = dwell + robot.control_dt if state["error"] <= config["success_tolerance_m"] else 0.0
            if dwell >= config["success_hold_s"]:
                break
        moved = float(np.linalg.norm(state["base"][:2]))
        return {"distance_m": float(d), "bearing_deg": bearing, "height_m": z, "success": bool(dwell >= config["success_hold_s"]),
                "time_s": round(t, 2), "closest_m": round(float(best), 4), "final_error_m": round(float(state["error"]), 4),
                "base_moved_m": round(moved, 4)}
    finally:
        robot.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--distances", type=float, nargs=3, metavar=("FIRST", "LAST", "STEP"),
                        help="horizontal distances to try, LAST included (default 0.3 to 1.2 m in 0.1 m steps)")
    args = parser.parse_args()
    out = ROOT/args.out
    if out.exists():
        raise FileExistsError(f"Choose a new output folder: {out}")
    distances = DISTANCES
    if args.distances:
        first, last, step = args.distances
        distances = np.round(np.arange(first, last + step/2, step), 3)
    jobs = list(itertools.product(distances, BEARINGS, HEIGHTS))
    with Pool(args.workers) as pool:
        results = pool.map(attempt, jobs)
    out.mkdir(parents=True)
    (out/"results.json").write_text(json.dumps({"rule": "hand within 1 cm for 0.25 s, expert arm, base still",
                                               "duration_s": DURATION, "results": results}, indent=1))
    for bearing, z in itertools.product(BEARINGS, HEIGHTS):
        row = [r for r in results if r["bearing_deg"] == bearing and r["height_m"] == z]
        print(f"bearing {bearing:4.0f} deg, height {z:.2f} m: " +
              " ".join(f"{r['distance_m']:.2f}{'✓' if r['success'] else '✗'}" for r in row))


if __name__ == "__main__":
    main()
