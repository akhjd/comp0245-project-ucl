"""DAgger for the lidar base policy (Ross, Gordon and Bagnell, 2011).

Plain behaviour cloning only ever sees states the expert visits. Once the learned policy drifts somewhere the
expert never goes (e.g. stopped, facing away from the target), it has nothing to copy. DAgger fixes that:

    round 0: train on expert data (plain behaviour cloning)
    round k: let the round k-1 model DRIVE on training scenarios; at every step ask the expert what it would do
             there ("expert_wheels"; it never moves the robot); add those labelled states to the training data;
             retrain on everything collected so far
Every round's model is scored in closed loop on the same validation scenarios (eval_obstacles.score).

This works here because the base expert (obstacle_env.SteerAround) is memoryless and sensor-only: it can label
any state from the current sensors alone, with no history to replay.

Hard layouts (optional): --hard-scenarios adds a pool of expert-solved hard training layouts; each round then
drives half normal, half hard layouts, so normal driving keeps being corrected too. --select-scenarios scores every
round on a separate set of hard validation layouts (used to choose the round). --balance weights the loss so
normal and hard rows count equally.

Example (about 30 min):
    python dagger.py --name nov --no-velocity --rounds 3 \
        --data runs/obst-v2b runs/obst-v3-extra-0 runs/obst-v3-extra-1 runs/obst-v3-extra-2 runs/obst-v3-extra-3
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SAVE_EVERY = 10       # store 100 rows per second instead of 1000: about 10x less disk; training reads every 10th row
SCENARIO_FILES = [ROOT/"scenarios_obstacles_v2.json", ROOT/"scenarios_obstacles_v3_extra.json"]


def _episode(job):
    """One closed-loop episode in its own process (each process has its own simulator)."""
    model_path, scenario, folder, duration, label, perturb = job
    import contextlib
    import torch
    torch.set_num_threads(1)
    from eval_obstacles import load_model, LearnedBase, score
    from obstacle_env import rollout
    from pipeline import read_json
    model, saved = load_model(model_path)
    config = read_json(ROOT/"settings.json")
    config["duration_s"] = duration
    with open(os.devnull, "w") as quiet, contextlib.redirect_stderr(quiet):
        summary = rollout(config, scenario, folder, LearnedBase(model, saved),
                          label_expert=label, lost_distance=6.0 if label else None, save_every=SAVE_EVERY,
                          perturb=perturb)
    return summary, (None if label else score(Path(folder), scenario, summary))


def run_episodes(model_path, scenarios, out, duration, label, workers, noise=(0.0, 0.0), seed=0):
    """noise = (max lidar noise m, max dropout chance): each driven episode gets its own random strength."""
    out.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    perturbs = [{"lidar_noise": float(rng.uniform(0, noise[0])), "lidar_dropout": float(rng.uniform(0, noise[1])),
                 "seed": int(rng.integers(1_000_000))} if max(noise) > 0 else None for _ in scenarios]
    jobs = [(str(model_path), s, str(out/s["id"]), duration, label, p) for s, p in zip(scenarios, perturbs)]
    with mp.get_context("spawn").Pool(workers) as pool:
        results = pool.map(_episode, jobs, chunksize=1)
    manifest = {"saved_every": SAVE_EVERY, "episodes": [{"path": s["id"], "scenario_id": s["id"], "split": "train" if label else s["split"],
                              **summary} for s, (summary, _) in zip(scenarios, results)]}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=1))
    return [r for _, r in results]


def evaluate(model_path, out, validation, duration, workers):
    scores = run_episodes(model_path, validation, out, duration, False, workers)
    n = len(scores)
    totals = {"episodes": n, "strict": sum(r["strict"] for r in scores), "loose": sum(r["loose"] for r in scores),
              "no_contact": sum(r["contact_steps"] == 0 for r in scores),
              "reachable": sum(r["reachable"] for r in scores),
              "median_distance_error_m": float(np.median([r["distance_error_m"] for r in scores]))}
    (out/"evaluation.json").write_text(json.dumps({"model": str(model_path), "arm": "fixed PD hold (not learned)",
                                                   "totals": totals, "episodes": scores}, indent=1))
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="results go to runs/dagger-NAME")
    parser.add_argument("--data", nargs="+", required=True, help="expert dataset folders")
    parser.add_argument("--no-velocity", action="store_true")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--per-round", type=int, default=60, help="training scenarios driven per round")
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lidar-noise", type=float, default=0.0,
                        help="noise-robust DAgger: max lidar noise (m) the policy sees while driving (expert labels "
                             "use the true lidar) and for a noisy copy of the expert data")
    parser.add_argument("--lidar-dropout", type=float, default=0.0, help="max chance a ray drops out")
    parser.add_argument("--hard-scenarios", help="file of expert-solved hard training layouts (train split)")
    parser.add_argument("--select-scenarios", help="file of hard validation layouts scored every round")
    parser.add_argument("--hard-duration", type=float, default=60.0, help="seconds per hard-layout episode")
    parser.add_argument("--balance", action="store_true", help="hard and normal rows count equally in the loss")
    args = parser.parse_args()
    noise = (args.lidar_noise, args.lidar_dropout)
    from train_base_torch import train

    root = ROOT/"runs"/f"dagger-{args.name}"
    if root.exists():
        raise FileExistsError(f"Choose a new name: {root}")
    root.mkdir(parents=True)
    scenarios = [s for f in SCENARIO_FILES for s in json.loads(f.read_text())]
    pool = [s for s in scenarios if s["split"] == "train"]
    validation = [s for s in scenarios if s["split"] == "validation"]      # the test split is never touched
    hard_pool = ([s for s in json.loads(Path(args.hard_scenarios).read_text()) if s["split"] == "train"]
                 if args.hard_scenarios else [])
    select = ([s for s in json.loads(Path(args.select_scenarios).read_text()) if s["split"] == "validation"]
              if args.select_scenarios else [])
    rng = np.random.default_rng(args.seed)
    dagger_dirs, table = [], []
    for k in range(args.rounds + 1):
        if k > 0:   # let the previous model drive, with the expert labelling every state it visits
            n_hard = args.per_round // 2 if hard_pool else 0
            chosen = [pool[i] for i in rng.choice(len(pool), size=min(args.per_round - n_hard, len(pool)), replace=False)]
            data_dir = root/f"r{k}-data"
            run_episodes(root/f"r{k-1}"/"model.pt", chosen, data_dir, args.duration, True, args.workers,
                         noise, seed=args.seed + k)
            dagger_dirs.append(data_dir)
            if n_hard:
                chosen = [hard_pool[i] for i in rng.choice(len(hard_pool), size=min(n_hard, len(hard_pool)), replace=False)]
                data_dir = root/f"r{k}-hard-data"
                run_episodes(root/f"r{k-1}"/"model.pt", chosen, data_dir, args.hard_duration, True, args.workers,
                             noise, seed=args.seed + 100 + k)
                dagger_dirs.append(data_dir)
        rmse = train(args.data, dagger_dirs, root/f"r{k}", args.no_velocity, seed=args.seed,
                     lidar_noise=args.lidar_noise, lidar_dropout=args.lidar_dropout, balance=args.balance)
        totals = evaluate(root/f"r{k}"/"model.pt", root/f"r{k}-validation", validation, args.duration, args.workers)
        if select:
            hard_totals = evaluate(root/f"r{k}"/"model.pt", root/f"r{k}-select", select, args.hard_duration, args.workers)
            totals.update({f"select_{key}": v for key, v in hard_totals.items()})
        table.append({"round": k, "dagger_folders": len(dagger_dirs), "offline_rmse_rad_s": rmse, **totals})
        print(f"ROUND {k}: offline RMSE {rmse:.3f}, closed loop {totals}", flush=True)
        (root/"results.json").write_text(json.dumps({"no_velocity": args.no_velocity, "data": args.data,
                                                     "lidar_noise": args.lidar_noise, "lidar_dropout": args.lidar_dropout,
                                                     "hard_scenarios": args.hard_scenarios,
                                                     "select_scenarios": args.select_scenarios,
                                                     "balance": args.balance, "rounds": table}, indent=1))


if __name__ == "__main__":
    main()
