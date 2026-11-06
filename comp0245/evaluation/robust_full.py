"""Robustness of the WHOLE robot: the fully learned robot (learned base parks, learned arm reaches)
and the full-task expert, under the same adversities as comp0245/evaluation/robustness.py (which
tests the base alone, arm held still).

Success = the course rule (hand within 1 cm for 0.25 s) with no contact, as in
comp0245/evaluation/eval_arm.py. One level per test, the strongest from
comp0245/evaluation/robustness.py:
    clean   no adversity
    lidar   0.3 m Gaussian noise per ray + 30% dropouts (the controller sees the noisy lidar)
    push    120 N sideways for 0.3 s at t = 6 s (about 0.3 m), alternating left/right
    delay   wheel commands reach the motors 400 ms late (arm torques stay on time: the stiff arm PD
            becomes unstable with ~20 ms delay, see obstacle_env.rollout)
    hard    expert-solved hard layouts (--hard-file); the expert row there is 100% by construction
Scenarios: --validation-file (fresh normal layouts) for all tests except hard. No model is chosen
from this.

Example:
    python -m comp0245.evaluation.robust_full --name full --validation-file \\
        scenarios_obstacles_val_e.json \
        --hard-file scenarios_obstacles_hard3.json
"""

import argparse
import json
import multiprocessing as mp

import numpy as np

from comp0245.evaluation.eval_arm import _episode
from comp0245.paths import ROOT, scenario_path

TESTS = {
    "clean": {},
    "lidar": {"lidar_noise": 0.3, "lidar_dropout": 0.3},
    "push": {"push_n": 120},
    "delay": {"delay_ms": 400},
    "hard": {},
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", required=True, help="results go to runs/robust-NAME")
    parser.add_argument("--arm", default="weights/arm_final.pt")
    parser.add_argument("--base", default="weights/base_final.pt")
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--hard-file", required=True)
    parser.add_argument("--tests", nargs="*", default=list(TESTS))
    parser.add_argument("--no-expert", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--duration",
        type=float,
        default=70.0,
        help="seconds per episode (as for expert collection)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="only the first N scenarios per test (smoke tests)"
    )
    args = parser.parse_args()
    root = ROOT / "runs" / f"robust-{args.name}"
    if root.exists():
        raise FileExistsError(f"Choose a new name: {root}")
    root.mkdir(parents=True)
    validation = [
        s
        for s in json.loads(scenario_path(args.validation_file).read_text())
        if s["split"] == "validation"
    ]
    hard = [
        s
        for s in json.loads(scenario_path(args.hard_file).read_text())
        if s["split"] == "validation"
    ]
    if args.limit:
        validation, hard = validation[: args.limit], hard[: args.limit]
    controllers = [("fully_learned", str(ROOT / args.arm), str(ROOT / args.base))] + (
        [] if args.no_expert else [("expert", "expert", None)]
    )
    jobs, keys = [], []
    for label, arm, base in controllers:
        for test in args.tests:
            for i, s in enumerate(hard if test == "hard" else validation):
                pert = {**TESTS[test], "seed": i}
                if test == "push":
                    pert["push_sign"] = 1 if i % 2 == 0 else -1
                jobs.append(
                    (arm, base, s, str(root / label / test / s["id"]), args.duration, pert)
                )
                keys.append((label, test))
    print(f"{len(jobs)} episodes, {args.workers} workers", flush=True)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(_episode, jobs, chunksize=1)

    table = []
    for label, test in dict.fromkeys(keys):
        rs = [r for k, r in zip(keys, results, strict=False) if k == (label, test)]
        done = [r for r in rs if r["success"]]
        table.append(
            {
                "controller": label,
                "test": test,
                "perturb": TESTS[test],
                "episodes": len(rs),
                "success": len(done),
                "reached": sum(r["reached"] for r in rs),
                "contact": sum(r["contact_steps"] > 0 for r in rs),
                "never_parked": sum(r["parked_at_s"] is None for r in rs),
                "median_final_error_m": float(np.median([r["final_error_m"] for r in rs])),
                "median_time_s": float(
                    np.median([r["parked_at_s"] + r["reach_time_s"] for r in done])
                )
                if done
                else None,
            }
        )
        t = table[-1]
        print(
            f"{label:13s} {test:6s}: success {t['success']}/{t['episodes']}, contact "
            f"{t['contact']}, "
            f"never parked {t['never_parked']}, median error "
            f"{t['median_final_error_m'] * 1000:.1f} mm",
            flush=True,
        )
    (root / "results.json").write_text(
        json.dumps(
            {
                "arm": args.arm,
                "base": args.base,
                "duration_s": args.duration,
                "validation_file": args.validation_file,
                "hard_file": args.hard_file,
                "table": table,
                "episodes": [
                    {"controller": k[0], "test": k[1], **r}
                    for k, r in zip(keys, results, strict=False)
                ],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
