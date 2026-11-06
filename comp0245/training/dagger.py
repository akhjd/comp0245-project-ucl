"""DAgger for the lidar base policy (Ross, Gordon and Bagnell, 2011).

Plain behaviour cloning only ever sees states the expert visits. Once the learned policy drifts
somewhere the expert never goes (e.g. stopped, facing away from the target), it has nothing to
copy. DAgger fixes that:

    round 0: train on expert data (plain behaviour cloning)
    round k: let the round k-1 model DRIVE on training scenarios; at every step ask the expert what
             it would do there ("expert_wheels"; it never moves the robot); add those labelled
             states to the training data; retrain on everything collected so far
Every round's model is scored in closed loop on the same validation scenarios
(eval_obstacles.score).

This works here because the base expert (obstacle_env.SteerAround) is memoryless and sensor-only:
it can label any state from the current sensors alone, with no history to replay. --label-expert
rrt_visible labels with a shadow VisibleRRTExpert instead, which keeps its own plan beside the
policy (see obstacle_env.rollout).

Hard layouts (optional): --hard-scenarios adds a pool of expert-solved hard training layouts; each
round then drives half normal, half hard layouts, so normal driving keeps being corrected too.
--select-scenarios scores every round on a separate set of hard validation layouts (used to choose
the round). --balance weights the loss so normal and hard rows count equally.

Example (about 30 min):
    python -m comp0245.training.dagger --name nov --no-velocity --rounds 3 \
        --data runs/obst-v2b runs/obst-v3-extra-0 runs/obst-v3-extra-1 runs/obst-v3-extra-2
        runs/obst-v3-extra-3

Moving base in rooms, starting from a plain cloning model already trained and scored:
    python -m comp0245.training.dagger --name rooms-1scan --no-velocity --data runs/rooms-expert \\
        --validation-data runs/rooms-expert-val \
        --scenarios scenarios_rooms_train.json scenarios_rooms_val.json --label-expert rrt_visible
        \
        --start runs/scans-rooms/scans1-seed0 --duration 70 --workers 10
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
from pathlib import Path

import numpy as np

from comp0245.paths import ROOT, SCENARIO_DIR, SETTINGS_FILE, scenario_path

# store 100 rows per second instead of 1000: about 10x less disk; training reads every 10th row
SAVE_EVERY = 10
SCENARIO_FILES = [
    SCENARIO_DIR / "scenarios_obstacles_v2.json",
    SCENARIO_DIR / "scenarios_obstacles_v3_extra.json",
]
CONTACT_LIMIT_S, STALL_LIMIT_S = (
    2.0,
    20.0,
)  # labelled rooms runs end when stuck like this (Julia's values)


def _episode(job):
    """One closed-loop episode in its own process (each process has its own simulator)."""
    model_path, scenario, folder, duration, label, perturb, limits, rays = job
    import contextlib

    import torch

    torch.set_num_threads(1)
    from comp0245.evaluation.eval_obstacles import LearnedBase, load_model, score
    from comp0245.pipeline import read_json
    from comp0245.sim.obstacle_env import rollout

    model, saved = load_model(model_path)
    config = read_json(SETTINGS_FILE)
    config["duration_s"] = duration
    config["lidar_rays"] = rays or saved.get(
        "rays", 36
    )  # default: the lidar the driving model was trained on
    config["lidar_range"] = saved.get(
        "lidar_range", 4.0
    )  # the model's own lidar range (5 Nov; older models: 4 m)
    policy = LearnedBase(model, saved)
    try:
        with open(os.devnull, "w") as quiet, contextlib.redirect_stderr(quiet):
            summary = rollout(
                config,
                scenario,
                folder,
                policy,
                label_expert=label,
                lost_distance=6.0 if label else None,
                save_every=SAVE_EVERY,
                perturb=perturb,
                contact_limit_s=limits[0],
                stall_limit_s=limits[1],
            )
        policy.save_modes(folder)
    except Exception as error:
        if not label:  # scoring must never lose a run; one lost labelling run only costs its rows
            raise
        print(f"{scenario['id']}: labelling run failed: {error!r}", flush=True)
        return None, None
    return summary, (None if label else score(Path(folder), scenario, summary))


def run_episodes(
    model_path,
    scenarios,
    out,
    duration,
    label,
    workers,
    noise=(0.0, 0.0),
    seed=0,
    limits=(None, None),
    rays=None,
):
    """noise = (max lidar noise m, max dropout chance): each driven episode gets its own random
    strength. rays: lidar rays in the environment (None: the driving model's own ray count)."""
    out.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    perturbs = [
        {
            "lidar_noise": float(rng.uniform(0, noise[0])),
            "lidar_dropout": float(rng.uniform(0, noise[1])),
            "seed": int(rng.integers(1_000_000)),
        }
        if max(noise) > 0
        else None
        for _ in scenarios
    ]
    jobs = [
        (str(model_path), s, str(out / s["id"]), duration, label, p, limits, rays)
        for s, p in zip(scenarios, perturbs, strict=False)
    ]
    with mp.get_context("spawn").Pool(workers) as pool:
        results = pool.map(_episode, jobs, chunksize=1)
    import torch

    saved = torch.load(model_path, map_location="cpu", weights_only=False)
    rays = rays or saved.get("rays", 36)
    manifest = {
        "saved_every": SAVE_EVERY,
        "lidar_rays": rays,
        "lidar_range": saved.get("lidar_range", 4.0),
        "episodes": [
            {
                "path": s["id"],
                "scenario_id": s["id"],
                "split": "train" if label else s["split"],
                **summary,
            }
            for s, (summary, _) in zip(scenarios, results, strict=False)
            if summary is not None
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return [r for _, r in results]


def evaluate(model_path, out, validation, duration, workers):
    scores = run_episodes(model_path, validation, out, duration, False, workers)
    n = len(scores)
    totals = {
        "episodes": n,
        "parked": sum(r.get("parked", False) for r in scores),
        "strict": sum(r["strict"] for r in scores),
        "loose": sum(r["loose"] for r in scores),
        "no_contact": sum(r["contact_steps"] == 0 for r in scores),
        "reachable": sum(r["reachable"] for r in scores),
        "median_distance_error_m": float(np.median([r["distance_error_m"] for r in scores])),
    }
    (out / "evaluation.json").write_text(
        json.dumps(
            {
                "model": str(model_path),
                "arm": "fixed PD hold (not learned)",
                "totals": totals,
                "episodes": scores,
            },
            indent=1,
        )
    )
    return totals


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", required=True, help="results go to runs/dagger-NAME")
    parser.add_argument("--data", nargs="+", required=True, help="expert dataset folders")
    parser.add_argument("--no-velocity", action="store_true")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--per-round", type=int, default=60, help="training scenarios driven per round"
    )
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--lidar-noise",
        type=float,
        default=0.0,
        help="noise-robust DAgger: max lidar noise (m) the policy sees while driving (expert "
        "labels "
        "use the true lidar) and for a noisy copy of the expert data",
    )
    parser.add_argument(
        "--lidar-dropout", type=float, default=0.0, help="max chance a ray drops out"
    )
    parser.add_argument(
        "--hard-scenarios", help="file of expert-solved hard training layouts (train split)"
    )
    parser.add_argument(
        "--select-scenarios", help="file of hard validation layouts scored every round"
    )
    parser.add_argument(
        "--hard-duration", type=float, default=60.0, help="seconds per hard-layout episode"
    )
    parser.add_argument(
        "--balance", action="store_true", help="hard and normal rows count equally in the loss"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        help="scenario files for driving and scoring (default: the open floor sets)",
    )
    parser.add_argument(
        "--label-expert",
        choices=["rrt", "rrt_visible", "rrt_steady", "rrt_best", "window", "window_fast"],
        help="label with this shadow expert, not SteerAround",
    )
    parser.add_argument(
        "--validation-data",
        help="dataset for the validation loss (default as comp0245/training/train_base_torch.py)",
    )
    parser.add_argument("--scans", type=int, default=1, help="lidar scans per input")
    parser.add_argument("--hidden", type=int, default=64, help="neurons per hidden layer")
    parser.add_argument(
        "--encoder", choices=["mlp", "conv"], default="mlp", help="conv: learned ray filter"
    )
    parser.add_argument("--threat", action="store_true", help="training-only threat output")
    parser.add_argument(
        "--mixture", type=int, default=0, metavar="K", help="K candidate wheel commands (0: one)"
    )
    parser.add_argument(
        "--start", help="folder with a trained model and its validation/ scores: used as round 0"
    )
    parser.add_argument(
        "--lidar-rays",
        type=int,
        help="lidar rays (default: from the expert data's manifest); every "
        "dataset must match, and the environment is given the driving model's ray count",
    )
    parser.add_argument(
        "--modes", action="store_true", help="target and evade wheel heads with a learned switch"
    )
    parser.add_argument(
        "--mode-label",
        choices=["collision", "expert"],
        default="collision",
        help="evade label source",
    )
    parser.add_argument(
        "--arm-fold",
        action="store_true",
        help="arm folded inside the base outline while driving (scenario arm_q0)",
    )
    parser.add_argument(
        "--extra-dagger",
        nargs="*",
        default=[],
        help="DAgger data folders from earlier runs, trained on from round 1",
    )
    parser.add_argument(
        "--menu",
        action="store_true",
        help="pick one of the window teacher's 63 options + a correction",
    )
    parser.add_argument(
        "--evade-weight",
        type=float,
        default=1.0,
        help="loss weight of rows labelled evade (--mode-label)",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="training shows a random half of every batch left-right mirrored",
    )
    parser.add_argument(
        "--patience",
        type=int,
        help="stop training after this many epochs without a better validation loss",
    )
    parser.add_argument(
        "--balance-groups",
        action="store_true",
        help="every room type (scenario id prefix) weighs the same",
    )
    parser.add_argument(
        "--target-ray",
        action="store_true",
        help="'target blocked?' inputs (train_base_torch.target_ray)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="dropout after both trunk layers"
    )
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay")
    parser.add_argument(
        "--stride-ms",
        type=int,
        default=10,
        help="ms between training rows (train_base_torch.set_stride)",
    )
    parser.add_argument(
        "--validate-last",
        action="store_true",
        help="score only the last round on the validation scenarios (earlier rounds can be scored "
        "later)",
    )
    args = parser.parse_args()
    noise = (args.lidar_noise, args.lidar_dropout)
    from comp0245.training.train_base_torch import dataset_rays, set_stride, train

    set_stride(args.stride_ms)
    folder_rays = {
        f: dataset_rays(ROOT / f)
        for f in args.data + ([args.validation_data] if args.validation_data else [])
    }
    if args.lidar_rays and set(folder_rays.values()) != {args.lidar_rays}:
        raise SystemExit(
            f"--lidar-rays {args.lidar_rays} does not match the datasets: {folder_rays}"
        )

    root = ROOT / "runs" / f"dagger-{args.name}"
    if root.exists():
        raise FileExistsError(f"Choose a new name: {root}")
    root.mkdir(parents=True)
    files = [scenario_path(f) for f in args.scenarios] if args.scenarios else SCENARIO_FILES
    scenarios = [s for f in files for s in json.loads(f.read_text())]
    pool = [s for s in scenarios if s["split"] == "train"]
    limits = (None, None)
    if args.label_expert:
        pool = [{**s, "expert": args.label_expert} for s in pool]
        limits = (CONTACT_LIMIT_S, STALL_LIMIT_S)
    extra = {
        "scans": args.scans,
        "hidden": args.hidden,
        "encoder": args.encoder,
        "threat": args.threat,
        "mixture": args.mixture,
        "modes": args.modes,
        "mode_label": args.mode_label,
        "menu": args.menu,
        "evade_weight": args.evade_weight,
        "mirror": args.mirror,
        "patience": args.patience,
        "balance_groups": args.balance_groups,
        "target_ray": args.target_ray,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
    }
    if args.validation_data:
        extra["validation"] = args.validation_data
    validation = [
        s for s in scenarios if s["split"] == "validation"
    ]  # the test split is never touched
    hard_pool = (
        [
            s
            for s in json.loads(scenario_path(args.hard_scenarios).read_text())
            if s["split"] == "train"
        ]
        if args.hard_scenarios
        else []
    )
    if args.label_expert:
        hard_pool = [{**s, "expert": args.label_expert} for s in hard_pool]
    if args.arm_fold:
        from comp0245.evaluation.eval_obstacles import ARM_FOLD

        pool, hard_pool, validation = (
            [{**s, "arm_q0": ARM_FOLD} for s in group] for group in (pool, hard_pool, validation)
        )
    select = (
        [
            s
            for s in json.loads(scenario_path(args.select_scenarios).read_text())
            if s["split"] == "validation"
        ]
        if args.select_scenarios
        else []
    )
    rng = np.random.default_rng(args.seed)
    dagger_dirs, table = [ROOT / f for f in args.extra_dagger], []
    for k in range(args.rounds + 1):
        if k > 0:  # let the previous model drive, with the expert labelling every state it visits
            n_hard = args.per_round // 2 if hard_pool else 0
            chosen = [
                pool[i]
                for i in rng.choice(
                    len(pool), size=min(args.per_round - n_hard, len(pool)), replace=False
                )
            ]
            data_dir = root / f"r{k}-data"
            run_episodes(
                root / f"r{k - 1}" / "model.pt",
                chosen,
                data_dir,
                args.duration,
                True,
                args.workers,
                noise,
                seed=args.seed + k,
                limits=limits,
            )
            dagger_dirs.append(data_dir)
            if n_hard:
                chosen = [
                    hard_pool[i]
                    for i in rng.choice(
                        len(hard_pool), size=min(n_hard, len(hard_pool)), replace=False
                    )
                ]
                data_dir = root / f"r{k}-hard-data"
                run_episodes(
                    root / f"r{k - 1}" / "model.pt",
                    chosen,
                    data_dir,
                    args.hard_duration,
                    True,
                    args.workers,
                    noise,
                    seed=args.seed + 100 + k,
                    limits=limits,
                )
                dagger_dirs.append(data_dir)
        if k == 0 and args.start:
            start = ROOT / args.start
            shutil.copytree(start, root / "r0", ignore=shutil.ignore_patterns("validation"))
            rmse = json.loads((start / "training.json").read_text())["validation_wheel_rmse_rad_s"]
            totals = json.loads((start / "validation" / "evaluation.json").read_text())["totals"]
        else:
            rmse = train(
                args.data,
                dagger_dirs,
                root / f"r{k}",
                args.no_velocity,
                seed=args.seed,
                lidar_noise=args.lidar_noise,
                lidar_dropout=args.lidar_dropout,
                balance=args.balance,
                **extra,
            )
            if args.validate_last and k < args.rounds:  # scored later, in free time
                totals = {"validated": False}
            else:
                totals = evaluate(
                    root / f"r{k}" / "model.pt",
                    root / f"r{k}-validation",
                    validation,
                    args.duration,
                    args.workers,
                )
        if select and totals.get("validated", True):
            hard_totals = evaluate(
                root / f"r{k}" / "model.pt",
                root / f"r{k}-select",
                select,
                args.hard_duration,
                args.workers,
            )
            totals.update({f"select_{key}": v for key, v in hard_totals.items()})
        table.append(
            {"round": k, "dagger_folders": len(dagger_dirs), "offline_rmse_rad_s": rmse, **totals}
        )
        print(f"ROUND {k}: offline RMSE {rmse:.3f}, closed loop {totals}", flush=True)
        (root / "results.json").write_text(
            json.dumps(
                {
                    "no_velocity": args.no_velocity,
                    "data": args.data,
                    "lidar_noise": args.lidar_noise,
                    "lidar_dropout": args.lidar_dropout,
                    "hard_scenarios": args.hard_scenarios,
                    "select_scenarios": args.select_scenarios,
                    "balance": args.balance,
                    "label_expert": args.label_expert,
                    "scans": args.scans,
                    "hidden": args.hidden,
                    "encoder": args.encoder,
                    "threat": args.threat,
                    "mixture": args.mixture,
                    "modes": args.modes,
                    "mode_label": args.mode_label
                    if args.modes or args.evade_weight != 1
                    else None,
                    "menu": args.menu,
                    "evade_weight": args.evade_weight,
                    "arm_fold": args.arm_fold,
                    "extra_dagger": args.extra_dagger,
                    "mirror": args.mirror,
                    "patience": args.patience,
                    "balance_groups": args.balance_groups,
                    "stride_ms": args.stride_ms,
                    "target_ray": args.target_ray,
                    "dropout": args.dropout,
                    "weight_decay": args.weight_decay,
                    "validate_last": args.validate_last,
                    "lidar_rays": sorted(set(folder_rays.values())),
                    "rounds": table,
                },
                indent=1,
            )
        )


if __name__ == "__main__":
    main()
