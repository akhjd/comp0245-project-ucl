"""Drive the wheels MLP (inference only) on chosen scenarios, optionally in the PyBullet GUI, and
print how each run ended. Like the DAgger driving test: the MLP drives the base, the expert
controls the arm. Nothing is trained or labelled.

    python -m comp0245.apps.drive_mlp --case case3 --gui                                 floor \\
        plan, walls and furniture only
    python -m comp0245.apps.drive_mlp --case case4 --static-only --gui                   floor \\
        plan with still cylinders only
    python -m comp0245.apps.drive_mlp --case case3 case4 --static-only --count 10        10 runs \\
        without the GUI, then a summary
    python -m comp0245.apps.drive_mlp --id case4_0123 --gui                              one \\
        chosen scenario
    python -m comp0245.apps.drive_mlp --scenario-file runs/interactive/my_test.json --gui  \\
        scenarios made in comp0245/apps/interactive_test.py
    python -m comp0245.apps.drive_mlp --case case4 --static-only --split test --count 20 the \\
        test split (keep it for the final result)

Runs are saved in --out/<scenario id>/ (trajectory.npz, summary.json), so
comp0245/apps/view_runs.py can replay them.
"""

import argparse
import collections
import json
import shutil
from pathlib import Path

import numpy as np

from comp0245.paths import SETTINGS_FILE, scenario_path
from comp0245.training.dagger_mlp import (
    CONTACT_LIMIT_S,
    CONTROL_EVERY,
    STALL_LIMIT_S,
    LearnedWheels,
    scenarios_of,
)
from comp0245.training.train_mlp import CASES, load_policy


def moving(scenario):
    return any(np.any(np.asarray(o.get("vel", [0, 0])) != 0) for o in scenario["obstacles"])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="runs/dagger-pre/r3/best.pt")
    parser.add_argument("--data", default="runs/dataset")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--case", nargs="+", default=["case3"], choices=CASES)
    parser.add_argument(
        "--static-only", action="store_true", help="only scenarios with no moving cylinder"
    )
    parser.add_argument(
        "--id", nargs="+", help="these scenario ids (any split), instead of --case"
    )
    parser.add_argument(
        "--scenario-file",
        help="a JSON list of scenarios (such as from comp0245/apps/interactive_test.py), instead "
        "of --case",
    )
    parser.add_argument("--count", type=int, default=1, help="how many scenarios to drive")
    parser.add_argument(
        "--start", type=int, default=0, help="skip this many matching scenarios first"
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--out", default="runs/drive")
    args = parser.parse_args()

    if args.scenario_file:
        chosen = json.loads(scenario_path(args.scenario_file).read_text())
    elif args.id:
        every = [
            s
            for split in ("train", "validation", "test")
            for c in scenarios_of(args.data, split).values()
            for s in c
        ]
        chosen = [s for s in every if s["id"] in args.id]
    else:
        found = scenarios_of(args.data, args.split)
        chosen = [s for c in args.case for s in found[c] if not (args.static_only and moving(s))]
        chosen = chosen[args.start : args.start + args.count]
    if not chosen:
        raise SystemExit("No scenarios match")

    import torch

    from comp0245.pipeline import read_json
    from comp0245.sim.obstacle_env import rollout

    torch.set_num_threads(1)
    config = read_json(SETTINGS_FILE)
    config["duration_s"] = args.duration
    model, saved = load_policy(args.model)
    results = []
    for s in chosen:
        folder = Path(args.out) / s["id"]
        shutil.rmtree(folder, ignore_errors=True)
        cylinders = len(s["obstacles"])
        print(
            f"{s['id']} ({s['split']}): {len(s.get('targets', [s.get('target')]))} dot(s), "
            f"{cylinders} cylinder(s)"
            f"{', some moving' if moving(s) else ''}, "
            f"{sum(b.get('kind') == 'furniture' for b in s.get('boxes', []))} furniture",
            flush=True,
        )
        r = rollout(
            config,
            s,
            folder,
            gui=args.gui,
            save_every=CONTROL_EVERY,
            learned_wheels=LearnedWheels(model, saved),
            contact_limit_s=CONTACT_LIMIT_S,
            stall_limit_s=STALL_LIMIT_S,
        )
        results.append(r)
        print(
            f"  -> {r['termination']}, clean {r['success']}, dots "
            f"{r['dots_picked']}/{r['dots_total']}, "
            f"contact steps {r['contact_steps']}, {r['simulation_time_s']:.1f} s",
            flush=True,
        )
    clean = sum(r["success"] for r in results)
    print(
        f"\n{len(results)} runs with {args.model}: clean {clean}/{len(results)}, "
        f"ended {dict(collections.Counter(r['termination'] for r in results))}. Saved in "
        f"{args.out}/"
    )
    (Path(args.out) / "results.json").write_text(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
