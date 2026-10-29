"""Robustness suite for base controllers: how do they cope when the world is not perfect?

Runs each controller (the expert as reference, and any learned base models) on the validation scenarios under
each adversity at increasing strength, and scores the base exactly like eval_obstacles.py. No retraining: this
tests the models as they are. The reference is "expert_hold" (eval_obstacles.ExpertHold): the expert's wheels with
the same fixed arm hold and 40 s episode as the learned models. The full-task expert (arm reaches, episode ends
early) is not a like-for-like base reference; run it only on purpose with --models expert_full=expert. Tests:
    clean        no adversity (reference row)
    lidar        Gaussian noise per ray + random dropouts ("nothing there"); the controller sees the noisy lidar
    push         a 0.3 s sideways shove at t = 6 s (up to ~0.3 m), alternating left/right between scenarios
    delay        wheel commands reach the motors late (arm torques stay on time; see obstacle_env.rollout)
    hard         harder layouts never seen in training (scenarios_obstacles_hard.json): up to 7 bigger obstacles,
                 longer distances
Scenarios: --n of scenarios_obstacles_val100.json from --start (fresh validation scenarios never used to choose a
model), and all validation scenarios of --hard-file (default the original 30 hard layouts).
Contacts and failures are reported per level. The learned models' arm is held by a fixed PD (not learned).

Example:
    python robustness.py --name base-r2 --models nov=runs/dagger-nov/r2/model.pt vel=runs/dagger-vel2/r2/model.pt
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
TESTS = {
    "clean": [{}],
    "lidar": [{"lidar_noise": 0.05, "lidar_dropout": 0.05}, {"lidar_noise": 0.1, "lidar_dropout": 0.1},
              {"lidar_noise": 0.2, "lidar_dropout": 0.2}, {"lidar_noise": 0.3, "lidar_dropout": 0.3}],
    # measured in open space: 60 / 90 / 120 N for 0.3 s shove the base about 0.08 / 0.18 / 0.30 m sideways
    "push": [{"push_n": 60}, {"push_n": 90}, {"push_n": 120}],
    "delay": [{"delay_ms": 100}, {"delay_ms": 200}, {"delay_ms": 400}],
    "hard": [{}],
}
LEVEL_KEY = {"clean": None, "lidar": "lidar_noise", "push": "push_n", "delay": "delay_ms", "hard": None}


def _episode(job):
    model_path, scenario, folder, duration, perturb = job
    import contextlib
    import torch
    torch.set_num_threads(1)
    from eval_obstacles import load_model, LearnedBase, ExpertHold, score
    from obstacle_env import rollout
    from pipeline import read_json
    policy = None                    # "expert": the full-task expert
    if model_path == "expert-hold":
        policy = ExpertHold(scenario["target"])
    elif model_path != "expert":
        model, saved = load_model(model_path)
        policy = LearnedBase(model, saved)
    config = read_json(ROOT/"settings.json")
    config["duration_s"] = duration
    with open(os.devnull, "w") as quiet, contextlib.redirect_stderr(quiet):
        summary = rollout(config, scenario, folder, policy, save_every=10, perturb=perturb)
    return score(Path(folder), scenario, summary)


def summarise(keys, scores):
    """One table row per (controller, test, level) from the per-episode scores; also printed."""
    table = []
    for (label, test, level) in dict.fromkeys(keys):
        rs = [r for k, r in zip(keys, scores) if k == (label, test, level)]
        pert = TESTS[test][level]
        table.append({"controller": label, "test": test, "level": pert.get(LEVEL_KEY[test]) if LEVEL_KEY[test] else 0,
                      "perturb": pert, "episodes": len(rs),
                      "strict": sum(r["strict"] for r in rs), "loose": sum(r["loose"] for r in rs),
                      "reachable": sum(r["reachable"] for r in rs),
                      "contact": sum(r["contact_steps"] > 0 for r in rs),
                      "median_distance_error_m": float(np.median([r["distance_error_m"] for r in rs])),
                      "min_clearance_m": float(min(r["min_clearance_m"] for r in rs))})
        t = table[-1]
        print(f"{label:11s} {test:6s} level {t['level']!s:5s}: loose {t['loose']}/{t['episodes']}, "
              f"strict {t['strict']}, contact {t['contact']}, reachable {t['reachable']}, "
              f"min clearance {t['min_clearance_m']:.2f}", flush=True)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="results go to runs/robust-NAME")
    parser.add_argument("--models", nargs="*", default=[], help="label=path/to/model.pt ...")
    parser.add_argument("--no-expert", action="store_true", help="skip the expert_hold reference")
    parser.add_argument("--tests", nargs="*", default=list(TESTS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--limit", type=int, default=0, help="only the first N scenarios per test (smoke tests)")
    parser.add_argument("--n", type=int, default=30, help="how many of the 100 fresh validation scenarios to use")
    parser.add_argument("--start", type=int, default=0, help="skip the first N fresh validation scenarios")
    parser.add_argument("--validation-file", default=str(ROOT/"scenarios_obstacles_val100.json"))
    parser.add_argument("--hard-file", default=str(ROOT/"scenarios_obstacles_hard.json"))
    args = parser.parse_args()
    root = ROOT/"runs"/f"robust-{args.name}"
    if root.exists():
        raise FileExistsError(f"Choose a new name: {root}")
    root.mkdir(parents=True)
    controllers = (([] if args.no_expert else [("expert_hold", "expert-hold")])
                   + [tuple(m.split("=", 1)) for m in args.models])
    validation = [s for s in json.loads(Path(args.validation_file).read_text())
                  if s["split"] == "validation"][args.start:args.start + args.n]
    hard = [s for s in json.loads(Path(args.hard_file).read_text()) if s["split"] == "validation"]
    if args.limit:
        validation, hard = validation[:args.limit], hard[:args.limit]
    jobs, keys = [], []
    for label, path in controllers:
        for test in args.tests:
            for level, base_pert in enumerate(TESTS[test]):
                for i, s in enumerate(hard if test == "hard" else validation):
                    pert = {**base_pert, "seed": i}
                    if test == "push":
                        pert["push_sign"] = 1 if i % 2 == 0 else -1
                    jobs.append((path, s, str(root/label/f"{test}-{level}"/s["id"]), args.duration, pert))
                    keys.append((label, test, level))
    print(f"{len(jobs)} episodes, {args.workers} workers", flush=True)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        scores = pool.map(_episode, jobs, chunksize=1)
    table = summarise(keys, scores)
    (root/"results.json").write_text(json.dumps({"controllers": dict(controllers), "tests": TESTS,
                                                 "table": table, "episodes": [dict(zip(("controller", "test", "level"), k), **r)
                                                                              for k, r in zip(keys, scores)]}, indent=1))


if __name__ == "__main__":
    main()
