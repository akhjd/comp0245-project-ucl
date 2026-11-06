"""Train base models that differ only in how many lidar scans they see, and score each by driving.

Same data, network and seeds; the arm is held still, so this scores the base alone
(eval_obstacles.score). "Moved" counts runs that travelled at least 0.5 m: a model that copies its
own standstill fails this first. Results: runs/NAME/results.json.

Example:
    python -m comp0245.evaluation.compare_scans --name scans-rooms --data runs/rooms-expert \\
        --validation-data runs/rooms-expert-val \
        --scenarios scenarios_rooms_val.json --scans 1 3 --seeds 0 1 --workers 10
"""

import argparse
import json

import numpy as np

from comp0245.paths import ROOT, scenario_path
from comp0245.training.dagger import evaluate
from comp0245.training.train_base_torch import HIDDEN, train


def contacts(folder):
    """How many runs touched a mover, a static obstacle and a wall, and how many moved at all."""
    episodes = json.loads((folder / "manifest.json").read_text())["episodes"]
    mover, static = 0, 0
    for e in episodes:
        hits = np.array(e.get("contact_steps_per_obstacle", []), bool)
        moving = np.array(e.get("obstacle_is_moving", []), bool)
        mover += bool((hits & moving).any())
        static += bool((hits & ~moving).any())
    return {
        "mover_contact_runs": mover,
        "static_contact_runs": static,
        "wall_contact_runs": sum(e.get("wall_contact_steps", 0) > 0 for e in episodes),
        "moved_runs": sum(e["base_path_length_m"] >= 0.5 for e in episodes),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", required=True, help="results go to runs/NAME")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--scans", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--duration", type=float, default=70.0)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    root = ROOT / "runs" / args.name
    validation = [
        s
        for s in json.loads(scenario_path(args.scenarios).read_text())
        if s["split"] == "validation"
    ]
    table = []
    for scans in args.scans:
        for seed in args.seeds:
            folder = root / f"scans{scans}-seed{seed}"
            rmse = train(
                args.data,
                out=folder,
                no_velocity=True,
                seed=seed,
                validation=args.validation_data,
                scans=scans,
                hidden=args.hidden,
            )
            totals = evaluate(
                folder / "model.pt", folder / "validation", validation, args.duration, args.workers
            )
            table.append(
                {
                    "scans": scans,
                    "seed": seed,
                    "hidden": args.hidden,
                    "offline_rmse_rad_s": rmse,
                    **totals,
                    **contacts(folder / "validation"),
                }
            )
            print(table[-1], flush=True)
            (root / "results.json").write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
