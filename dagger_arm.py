"""DAgger for the learned arm, with a shadow expert as the teacher.

Model A copies the expert well while it is on the expert's path, but once it hovers a little off the target (states
the expert never visits: the expert succeeds and its episode ends) small errors pile up and the arm drifts away:
the same compounding error the base had. DAgger fixes it the same way:

    round 0: train on expert data (train_arm.py; the same as model A)
    round k: the round k-1 arm reaches on training scenarios (the expert's wheels park first); at every step a
             SHADOW copy of the course arm expert says what torque it would command there ("expert_arm",
             obstacle_env.rollout label_arm); retrain on the expert data plus every labelled round
Every round is scored in closed loop on the eval_arm.py scenarios (held-out episodes + fresh normal ones).

The arm expert has memory (its reference pose q_des), so it cannot simply be asked "what would you do here?".
The shadow keeps its own q_des, updated from the real arm every step; the expert caps q_des at 0.15 rad from the
measured angles, so the shadow stays anchored to where the arm really is. Check: with the expert itself
driving, the shadow's torques equal the expert's exactly (29 Oct: 0.0 N m difference over 20,090 steps).

With --base, a learned base parks instead of the expert's wheels, so the arm practises from the base's real
(slightly off) parkings; --init-model / --init-dagger continue from an earlier run's arm and data. Each round is
then scored twice: fully learned (that base) and with expert parking (to check it does not regress).

Example (about 1 hour):
    python dagger_arm.py --name a --rounds 3
    python dagger_arm.py --name b --rounds 3 --base runs/dagger-novnoise-hard/r3/model.pt \
        --init-model runs/dagger-arm-a/r5/model.pt --init-dagger runs/dagger-arm-a/r{1,2,3,4,5}-data
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
from pathlib import Path
import numpy as np
from eval_arm import ARM_DATA, run_eval

ROOT = Path(__file__).resolve().parent
SAVE_EVERY = 10    # 100 rows per second; train_arm.py reads every 10th 1 kHz row anyway


def _episode(job):
    arm_path, base_path, scenario, folder, duration = job
    import contextlib
    import torch
    torch.set_num_threads(1)
    from eval_arm import LearnedArm, load_arm
    from obstacle_env import rollout
    from pipeline import read_json
    config = read_json(ROOT/"settings.json")
    config["duration_s"] = duration
    base = None
    if base_path:
        from eval_obstacles import load_model, LearnedBase
        base = LearnedBase(*load_model(base_path))
    policy = LearnedArm(*load_arm(arm_path), scenario["target"], base)
    with open(os.devnull, "w") as quiet, contextlib.redirect_stderr(quiet):
        return rollout(config, scenario, folder, policy, label_arm=True, save_every=SAVE_EVERY)


def drive(arm_path, scenarios, out, duration, workers, base=None):
    """The arm drives every scenario; the shadow expert labels every step. Writes a manifest like a dataset's.
    base: a learned base model.pt parks instead of the expert's wheels."""
    out.mkdir(parents=True)
    jobs = [(str(arm_path), base, s, str(out/s["id"]), duration) for s in scenarios]
    with mp.get_context("spawn").Pool(workers) as pool:
        summaries = pool.map(_episode, jobs, chunksize=1)
    manifest = {"saved_every": SAVE_EVERY, "episodes": [{"path": s["id"], "scenario_id": s["id"], "split": "train",
                                                          **summary} for s, summary in zip(scenarios, summaries)]}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=1))


def training_pool():
    """Training scenarios of the arm datasets: never the held-out every-10th episodes that eval_arm.py scores."""
    pool = {}
    for folder, file in ARM_DATA.items():
        episodes = json.loads((ROOT/"runs"/folder/"manifest.json").read_text())["episodes"]
        keep = {e["scenario_id"] for i, e in enumerate(episodes) if i % 10 != 0 and e["success"]}
        pool[folder] = [s for s in json.loads((ROOT/file).read_text()) if s["id"] in keep]
    return pool


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="results go to runs/dagger-arm-NAME")
    parser.add_argument("--data", nargs="+", default=[f"runs/{f}" for f in ARM_DATA], help="expert arm datasets")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--per-round", type=int, default=60, help="scenarios driven per round (split evenly per set)")
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base", help="learned base model.pt that parks (default: the expert's wheels)")
    parser.add_argument("--init-model", help="round 0 = this arm model.pt (copied, not retrained)")
    parser.add_argument("--init-dagger", nargs="*", default=[], help="earlier arm DAgger folders to keep training on")
    parser.add_argument("--resume", action="store_true",
                        help="continue an existing run to more --rounds (finished rounds are kept, not redone)")
    args = parser.parse_args()
    from train_arm import train

    root = ROOT/"runs"/f"dagger-arm-{args.name}"
    if root.exists() and not args.resume:
        raise FileExistsError(f"Choose a new name: {root}")
    root.mkdir(parents=True, exist_ok=True)
    done = {r["round"]: r for r in json.loads((root/"results.json").read_text())["rounds"]} if args.resume else {}
    pool = training_pool()
    rng = np.random.default_rng(args.seed)
    dagger_dirs, table = list(args.init_dagger), []
    for k in range(args.rounds + 1):
        if k > 0:
            per_set = args.per_round // len(pool)
            chosen = [s for group in pool.values()   # drawn even for finished rounds, so later draws match a fresh run
                      for s in (group[i] for i in rng.choice(len(group), size=min(per_set, len(group)), replace=False))]
            if k not in done:
                drive(root/f"r{k-1}"/"model.pt", chosen, root/f"r{k}-data", args.duration, args.workers, args.base)
            dagger_dirs.append(str(root/f"r{k}-data"))
        if k in done:
            table.append(done[k])
            continue
        if k == 0 and args.init_model:
            (root/"r0").mkdir()
            shutil.copy(args.init_model, root/"r0"/"model.pt")
            (root/"r0"/"training.json").write_text(json.dumps({"copied_from": args.init_model}, indent=1))
        else:
            train(args.data, root/f"r{k}", seed=args.seed, dagger=dagger_dirs)
        model = str(root/f"r{k}"/"model.pt")
        row = {"round": k, "dagger_folders": len(dagger_dirs)}
        if args.base:     # fully learned first (the round is chosen on it), then expert parking
            full = run_eval(model, args.base, root/f"r{k}-eval-full", args.workers, args.duration)
            row.update({f"full_{t['set']}": t["success"] for t in full})
            row["full_success"] = sum(t["success"] for t in full)
        scores = run_eval(model, None, root/f"r{k}-eval", args.workers, args.duration)
        row.update({**{t["set"]: t["success"] for t in scores},
                    "success": sum(t["success"] for t in scores), "episodes": sum(t["episodes"] for t in scores)})
        table.append(row)
        print(f"ROUND {k}: {row}", flush=True)
        (root/"results.json").write_text(json.dumps({"data": args.data, "base": args.base or "expert wheels",
                                                     "init_model": args.init_model, "init_dagger": args.init_dagger,
                                                     "rounds": table}, indent=1))


if __name__ == "__main__":
    main()
