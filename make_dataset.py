"""Write the scenario files for the five expert data cases, with train / validation / test splits.

    case1  open room (walls only), no obstacles, 1 dot
    case2  open room, static and moving cylinders, 1 dot
    case3  random floor plan (interior walls, doorways, furniture), no cylinders, 1 dot
    case4  random floor plan, static and moving cylinders, 1 dot
    case5  random floor plan, static and moving cylinders, 3 dots

Example:
    python make_dataset.py --out runs/dataset
    python make_dataset.py --out runs/pilot --scale 0.05
"""
import argparse
import json
import types
from pathlib import Path
import numpy as np
import make_floorplan_scenarios as fp
import make_obstacle_scenarios as open_room
from obstacle_env import fit_to_room

COUNTS = {"case1": 150, "case2": 400, "case3": 300, "case4": 500, "case5": 300}
OPEN_ROOM_HALF = 5.0

def open_scenario(rng, obstacles):
    args = types.SimpleNamespace(distance=[2.8, 4.0], height=[0.35, 0.8], radius=[0.15, 0.3], speed=[0.1, 0.3],
                                 static_count=[0, 3] if obstacles else [0, 0], moving_count=[0, 2] if obstacles else [0, 0])
    while True:
        s = open_room.sample_moving_scenario(rng, args)
        if obstacles and not s["obstacles"]:
            continue
        try:
            return fit_to_room({**s, "id": ""}, OPEN_ROOM_HALF)[0]
        except ValueError:
            continue

def plan_scenario(rng, statics, mover_share, dots):
    args = types.SimpleNamespace(cross_rooms=0.7, mover_share=mover_share, dots=dots, height=[0.35, 0.8],
                                 static_count=statics, radius=[0.15, 0.3])
    return fp.sample(rng, "random", args)

def make(case, rng):
    if case == "case1":
        return open_scenario(rng, False)
    if case == "case2":
        return open_scenario(rng, True)
    if case == "case3":
        return plan_scenario(rng, [0, 0], 0.0, 1)
    while True:   # cases 4 and 5 always have at least one cylinder
        s = plan_scenario(rng, [0, 3], 0.7, 1 if case == "case4" else 3)
        if s["obstacles"]:
            return s

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scale", type=float, default=1.0, help="share of the full run counts")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--jobs", type=int, default=7, help="number of files each case is split into")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    for offset, (case, count) in enumerate(COUNTS.items()):
        rng = np.random.default_rng([args.seed, offset])
        n = max(6, int(round(count*args.scale)))
        scenarios = []
        for i in range(n):
            split = "test" if i % 20 == 19 else "validation" if i % 20 == 9 else "train"   # 90 / 5 / 5
            scenarios.append({**make(case, rng), "id": f"{case}_{i:04d}", "split": split, "case": case,
                              "seed": int(rng.integers(1_000_000)), "expert": "rrt"})
        (out/f"{case}.json").write_text(json.dumps(scenarios))
        for j in range(args.jobs):
            (out/f"{case}.part{j}.json").write_text(json.dumps(scenarios[j::args.jobs]))
        print(case, n, "scenarios,", sum(bool(s["obstacles"]) for s in scenarios), "with obstacles,",
              sum(any("vel" in o for o in s["obstacles"]) for s in scenarios), "with a mover", flush=True)

if __name__ == "__main__":
    main()
