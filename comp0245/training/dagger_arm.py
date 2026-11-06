"""DAgger for the learned arm, with a shadow expert as the teacher.

Model A copies the expert well while it is on the expert's path, but once it hovers a little off
the target (states the expert never visits: the expert succeeds and its episode ends) small errors
pile up and the arm drifts away:
the same compounding error the base had. DAgger fixes it the same way:

    round 0: train on expert data (comp0245/training/train_arm.py; the same as model A)
    round k: the round k-1 arm reaches on training scenarios (the expert's wheels park first); at
             every step a SHADOW copy of the course arm expert says what torque it would command
             there ("expert_arm", obstacle_env.rollout label_arm); retrain on the expert data plus
             every labelled round
Every round is scored in closed loop on the comp0245/evaluation/eval_arm.py scenarios (held-out
episodes + fresh normal ones).

The arm expert has memory (its reference pose q_des), so it cannot simply be asked "what would you
do here?". The shadow keeps its own q_des, updated from the real arm every step; the expert caps
q_des at 0.15 rad from the measured angles, so the shadow stays anchored to where the arm really
is. Check: with the expert itself driving, the shadow's torques equal the expert's exactly (29 Oct:
0.0 N m difference over 20,090 steps).

Example (about 1 hour):
    python -m comp0245.training.dagger_arm --name a --rounds 3
"""

import argparse
import json
import multiprocessing as mp
import os

import numpy as np

from comp0245.evaluation.eval_arm import ARM_DATA, run_eval
from comp0245.paths import ROOT, SETTINGS_FILE, scenario_path

SAVE_EVERY = (
    10  # 100 rows per second; comp0245/training/train_arm.py reads every 10th 1 kHz row anyway
)


def _episode(job):
    arm_path, scenario, folder, duration = job
    import contextlib

    import torch

    torch.set_num_threads(1)
    from comp0245.evaluation.eval_arm import LearnedArm, load_arm
    from comp0245.pipeline import read_json
    from comp0245.sim.obstacle_env import rollout

    config = read_json(SETTINGS_FILE)
    config["duration_s"] = duration
    policy = LearnedArm(*load_arm(arm_path), scenario["target"])
    with open(os.devnull, "w") as quiet, contextlib.redirect_stderr(quiet):
        return rollout(config, scenario, folder, policy, label_arm=True, save_every=SAVE_EVERY)


def drive(arm_path, scenarios, out, duration, workers):
    """The arm drives every scenario; the shadow expert labels every step. Writes a manifest
    like a dataset's."""
    out.mkdir(parents=True)
    jobs = [(str(arm_path), s, str(out / s["id"]), duration) for s in scenarios]
    with mp.get_context("spawn").Pool(workers) as pool:
        summaries = pool.map(_episode, jobs, chunksize=1)
    manifest = {
        "saved_every": SAVE_EVERY,
        "episodes": [
            {"path": s["id"], "scenario_id": s["id"], "split": "train", **summary}
            for s, summary in zip(scenarios, summaries, strict=False)
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


def training_pool():
    """Training scenarios of the arm datasets: never the held-out every-10th episodes that
    comp0245/evaluation/eval_arm.py scores."""
    pool = {}
    for folder, file in ARM_DATA.items():
        episodes = json.loads((ROOT / "runs" / folder / "manifest.json").read_text())["episodes"]
        keep = {e["scenario_id"] for i, e in enumerate(episodes) if i % 10 != 0 and e["success"]}
        pool[folder] = [s for s in json.loads(scenario_path(file).read_text()) if s["id"] in keep]
    return pool


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", required=True, help="results go to runs/dagger-arm-NAME")
    parser.add_argument(
        "--data", nargs="+", default=[f"runs/{f}" for f in ARM_DATA], help="expert arm datasets"
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--per-round",
        type=int,
        default=60,
        help="scenarios driven per round (split evenly per set)",
    )
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from comp0245.training.train_arm import train

    root = ROOT / "runs" / f"dagger-arm-{args.name}"
    if root.exists():
        raise FileExistsError(f"Choose a new name: {root}")
    root.mkdir(parents=True)
    pool = training_pool()
    rng = np.random.default_rng(args.seed)
    dagger_dirs, table = [], []
    for k in range(args.rounds + 1):
        if k > 0:
            per_set = args.per_round // len(pool)
            chosen = [
                s
                for group in pool.values()
                for s in (
                    group[i]
                    for i in rng.choice(len(group), size=min(per_set, len(group)), replace=False)
                )
            ]
            drive(
                root / f"r{k - 1}" / "model.pt",
                chosen,
                root / f"r{k}-data",
                args.duration,
                args.workers,
            )
            dagger_dirs.append(str(root / f"r{k}-data"))
        train(args.data, root / f"r{k}", seed=args.seed, dagger=dagger_dirs)
        scores = run_eval(
            str(root / f"r{k}" / "model.pt"),
            None,
            root / f"r{k}-eval",
            args.workers,
            args.duration,
        )
        row = {
            "round": k,
            "dagger_folders": len(dagger_dirs),
            **{t["set"]: t["success"] for t in scores},
            "success": sum(t["success"] for t in scores),
            "episodes": sum(t["episodes"] for t in scores),
        }
        table.append(row)
        print(f"ROUND {k}: {row}", flush=True)
        (root / "results.json").write_text(
            json.dumps({"data": args.data, "rounds": table}, indent=1)
        )


if __name__ == "__main__":
    main()
