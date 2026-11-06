"""DAgger for the MLP: it drives, the RRT expert labels every state it visits, and it is retrained
on everything so far.

    python -m comp0245.training.dagger_mlp --start runs/mlp1/best.pt --out runs/dagger1

Round 0 tests the starting model driving. Each later round lets the latest model drive --per-case
training scenarios of every case, retrains from scratch on the expert data plus all DAgger runs so
far, and tests the new model driving on the validation scenarios. Results go to results.json. Run
the same command again to continue after an interruption.

While the MLP controls the wheels, the expert controls the arm and decides when to reach (the
MLP outputs wheels only). The test split is not used. Runs where the MLP is stuck in contact for
2 s, or stops for 20 s, end early.
"""

import argparse
import collections
import glob
import json
import multiprocessing as mp
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from comp0245.paths import ROOT, SETTINGS_FILE
from comp0245.preprocess import FEATURE_SETS, Preprocessor, pose_of
from comp0245.training.train_mlp import (
    CASES,
    LIDAR_NAMES,
    OFFSETS,
    STATE_NAMES,
    Scale,
    load_policy,
    pre_version,
    state_rows,
)

CONTACT_LIMIT_S, STALL_LIMIT_S = 2.0, 20.0
CONTROL_EVERY = 10  # 1 ms steps per command: the MLP's training rows are 10 ms apart


class LearnedWheels:
    """The MLP as a wheel controller for rollout(learned_wheels=...). Make a new one for every
    episode."""

    def __init__(self, model, saved):
        self.model, self.scale = model, Scale(saved=saved)
        self.frames = collections.deque(maxlen=int(OFFSETS.max()) + 1)
        names = FEATURE_SETS[pre_version(self.scale.changes["door_memory"])]
        self.pre = (
            Preprocessor(names) if self.scale.changes["preprocess"] else None
        )  # fed every command, like the saved rows
        self.steps, self.wheels = 0, np.zeros(4)

    def __call__(self, state, robot, elapsed):
        if self.steps % CONTROL_EVERY == 0:
            import torch

            rows, _ = state_rows(
                state["x"][None],
                np.array([elapsed]),
                robot.dots,
                {p["dot"]: p["time_s"] for p in robot.picked},
            )
            if self.pre is not None:
                lidar = self.pre.step(
                    elapsed, state["lidar"], pose_of(state["x"]), rows[0][7:].reshape(-1, 4)
                )
            else:
                self.frames.append(np.asarray(state["lidar"], np.float32))
                newest = len(self.frames) - 1
                lidar = np.concatenate([self.frames[max(newest - k, 0)] for k in OFFSETS])
            x = self.scale.prepare(np.concatenate([lidar, rows[0]])[None])[0]
            with torch.no_grad():
                self.wheels = (
                    self.model(torch.from_numpy(x.astype(np.float32))[None])[0].numpy()
                    * self.scale.y_std
                    + self.scale.y_mean
                )
        self.steps += 1
        return self.wheels


def drive(job):
    """One episode in its own process, with its own simulator."""
    model_path, scenario, folder, duration = job
    import torch

    from comp0245.pipeline import read_json
    from comp0245.sim.obstacle_env import rollout

    torch.set_num_threads(1)
    summary_file = Path(folder) / "summary.json"
    if summary_file.exists():
        return json.loads(summary_file.read_text())
    shutil.rmtree(folder, ignore_errors=True)
    config = read_json(SETTINGS_FILE)
    config["duration_s"] = duration
    model, saved = load_policy(model_path)
    return rollout(
        config,
        scenario,
        folder,
        save_every=CONTROL_EVERY,
        learned_wheels=LearnedWheels(model, saved),
        contact_limit_s=CONTACT_LIMIT_S,
        stall_limit_s=STALL_LIMIT_S,
    )


def drive_all(model_path, scenarios, out, duration, workers):
    Path(out).mkdir(parents=True, exist_ok=True)
    jobs = [(str(model_path), s, str(Path(out) / s["id"]), duration) for s in scenarios]
    with mp.get_context("spawn").Pool(workers) as pool:
        return pool.map(drive, jobs, chunksize=1)


def score(model_path, scenarios, out, duration, workers):
    """Drive the validation scenarios and count, overall and per case: clean successes, dots
    touched, how runs ended."""
    done = Path(out) / "evaluation.json"
    if done.exists():
        return json.loads(done.read_text())
    begin = time.time()
    results = drive_all(model_path, scenarios, out, duration, workers)

    def tally(rows):
        good = [r for r in rows if r["success"]]
        return {
            "runs": len(rows),
            "success": len(good),
            "dots_touched": sum(r["dots_picked"] for r in rows),
            "dots_total": sum(r["dots_total"] for r in rows),
            "any_contact": sum(r["contact_steps"] > 0 for r in rows),
            "ended": dict(collections.Counter(r["termination"] for r in rows)),
            "mean_success_time_s": float(np.mean([r["simulation_time_s"] for r in good]))
            if good
            else None,
        }

    out_scores = {"all": tally(results), "seconds": time.time() - begin}
    for case in CASES:
        out_scores[case] = tally(
            [r for r, s in zip(results, scenarios, strict=False) if s["case"] == case]
        )
    done.write_text(json.dumps(out_scores, indent=1))
    return out_scores


def scenarios_of(data, split):
    """The scenarios of one split whose expert run was kept, by case, in a fixed order."""
    found = {c: [] for c in CASES}
    for name in sorted(glob.glob(f"{data}/case*/part*/*/trajectory.npz")):
        scenario = json.loads((Path(name).parent / "episode.json").read_text())["scenario"]
        if scenario["split"] == split:
            found[scenario["case"]].append(scenario)
    return found


def show(k, s):
    a = s["all"]
    print(
        f"ROUND {k} DRIVING: success {a['success']}/{a['runs']}, dots "
        f"{a['dots_touched']}/{a['dots_total']}, ended {a['ended']}, "
        f"{s['seconds'] / 60:.0f} min\n  "
        + ", ".join(f"{c} {s[c]['success']}/{s[c]['runs']}" for c in CASES),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--start", required=True, help="model to start from, such as runs/mlp1/best.pt"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default="runs/dataset")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--per-case", type=int, default=30, help="training scenarios driven per case in each round"
    )
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=list(CASES),
        choices=CASES,
        help="drive, train and test on these cases only",
    )
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    pool, validation = (
        scenarios_of(args.data, "train"),
        [
            s
            for c, v in scenarios_of(args.data, "validation").items()
            if c in args.cases
            for s in v
        ],
    )
    # A later round trains like the starting model, with the same input changes
    start = load_policy(args.start)[1]
    changes = start.get(
        "input_changes",
        {"no_velocity": len(start["input_names"]) < len(LIDAR_NAMES + STATE_NAMES)},
    )  # older models
    input_flags = ["--" + k.replace("_", "-") for k, on in changes.items() if on]
    # ...and the same regularisation and validation checks (older models: none)
    input_flags += [f"--{k.replace('_', '-')}={v}" for k, v in start.get("training", {}).items()]
    rng = np.random.default_rng(args.seed)
    order = {
        c: rng.permutation(len(pool[c])) for c in CASES
    }  # no scenario is driven twice while enough remain
    table = [
        {
            "round": 0,
            "model": args.start,
            **score(args.start, validation, root / "r0-validation", args.duration, args.workers),
        }
    ]
    show(0, table[0])
    (root / "results.json").write_text(json.dumps(table, indent=1))
    for k in range(1, args.rounds + 1):
        chosen = [
            pool[c][order[c][((k - 1) * args.per_case + i) % len(pool[c])]]
            for c in args.cases
            for i in range(args.per_case)
        ]
        begin = time.time()
        drive_all(table[-1]["model"], chosen, root / f"r{k}-data", args.duration, args.workers)
        print(
            f"ROUND {k}: drove {len(chosen)} runs in {(time.time() - begin) / 60:.0f} min",
            flush=True,
        )
        model_folder = root / f"r{k}"
        if not (model_folder / "result.json").exists():
            shutil.rmtree(model_folder, ignore_errors=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "comp0245.training.train_mlp",
                    "--data",
                    args.data,
                    "--out",
                    str(model_folder),
                    "--cases",
                    *args.cases,
                    *input_flags,
                    "--dagger",
                    *[str(root / f"r{j}-data") for j in range(1, k + 1)],
                ],
                check=True,
                cwd=ROOT,
            )
        table.append(
            {
                "round": k,
                "model": str(model_folder / "best.pt"),
                **score(
                    model_folder / "best.pt",
                    validation,
                    root / f"r{k}-validation",
                    args.duration,
                    args.workers,
                ),
            }
        )
        show(k, table[-1])
        (root / "results.json").write_text(json.dumps(table, indent=1))
    (root / "results.json").write_text(json.dumps(table, indent=1))
    best = max(table, key=lambda r: (r["all"]["success"], r["all"]["dots_touched"]))
    print(
        f"\nBest by driving on validation: round {best['round']}, {best['model']}: success "
        f"{best['all']['success']}/{best['all']['runs']}, "
        f"dots {best['all']['dots_touched']}/{best['all']['dots_total']}"
    )


if __name__ == "__main__":
    main()
