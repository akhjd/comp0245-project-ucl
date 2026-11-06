"""Check of the door memory (comp0245/preprocess.py feature set v2) against the scenario truth, on
saved expert runs. Not a unit test.

    python tests/check_door_memory.py                  15 runs per case for cases 3-5, writes \\
        runs/door_memory_check.json
    python tests/check_door_memory.py --runs 5

Measures, per case:
  v1_unchanged    the v1 features computed now equal the cached features_v1.npy (largest
                  difference)
  remembered      share of a run's true doorways that end up confirmed in the memory (recall)
  correct         share of confirmed doorways within 0.5 m of a true doorway (precision), and their
                  position error
  rows_with_door  share of rows where the MLP gets a remembered doorway (door_seen)
  across_right    rows where a dot's doorway is reported as "across" (dot on the other side of its
                  wall): share where the matching true wall really does separate robot and dot
  ms_per_scan     preprocessing time per scan, all features of v2
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))  # appended, so PYTHONPATH can point at another copy to check it first
from comp0245 import preprocess as pp  # noqa: E402
from comp0245.training.train_mlp import state_rows  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
from check_preprocess import true_doors  # noqa: E402

MATCH_M = 0.5


def check_run(folder, tally):
    z = np.load(folder / "trajectory.npz")
    summary = json.loads((folder / "summary.json").read_text())
    scenario = json.loads((folder / "episode.json").read_text())["scenario"]
    obs, t, lidar = z["observations"], z["time_s"], z["lidar"]
    targets = np.array(scenario.get("targets", [scenario["target"]]))
    slots, _ = state_rows(obs, t, targets, {p["dot"]: p["time_s"] for p in summary["picks"]})
    dots = slots[:, 7:].reshape(-1, 3, 4)
    doors = true_doors(scenario.get("boxes", []))
    names = pp.FEATURE_SETS["v2"]
    pre = pp.Preprocessor(names)
    begin = time.time()
    f = np.stack(
        [
            pre.step(ti, scan, pp.pose_of(o), d)
            for ti, scan, o, d in zip(t, lidar, obs, dots, strict=False)
        ]
    )
    tally["ms"].append(1000 * (time.time() - begin) / len(t))
    cache = folder / "features_v1.npy"
    if cache.exists():
        tally["v1_diff"] = max(
            tally["v1_diff"], float(np.abs(np.load(cache) - f[:, : len(pp.MLP_NAMES)]).max())
        )
    confirmed = [d["xy"] for d in pre.doors if pp.confirmed_door(d)]
    tally["true_doors"] += len(doors)
    tally["remembered"] += sum(
        any(np.linalg.norm(c - m) < MATCH_M for c in confirmed) for m, _ in doors
    )
    for c in confirmed:
        errors = [np.linalg.norm(c - m) for m, _ in doors]
        tally["confirmed"] += 1
        if errors and min(errors) < MATCH_M:
            tally["correct"] += 1
            tally["error_m"].append(min(errors))
    col = {n: i for i, n in enumerate(names)}
    tally["rows"] += len(t)
    tally["rows_with_door"] += int(f[:, col["door_seen"]].sum())
    yaw = np.arctan2(obs[:, 3], obs[:, 4])
    for k in range(3):
        rows = np.flatnonzero(f[:, col[f"dot{k}_door_across"]] > 0.5)
        for i in rows[::10]:  # every 10th row is plenty
            c, s = np.cos(yaw[i]), np.sin(yaw[i])
            d, si, co = (
                f[i, col[f"dot{k}_door_distance"]],
                f[i, col[f"dot{k}_door_sin"]],
                f[i, col[f"dot{k}_door_cos"]],
            )
            robot = obs[i, :2]
            door = robot + d * np.array([c * co - s * si, s * co + c * si])
            match = [(np.linalg.norm(door - m), m, along) for m, along in doors]
            if not match or min(match, key=lambda m: m[0])[0] >= MATCH_M:
                tally["across_no_true_door"] += 1
                continue
            _, m, along = min(match, key=lambda m: m[0])

            def side(p):
                return along[0] * (p[1] - m[1]) - along[1] * (p[0] - m[0])

            tally["across_checked"] += 1
            tally["across_right"] += int(side(robot) * side(targets[k][:2]) < 0)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default=str(ROOT / "runs/dataset"))
    parser.add_argument("--runs", type=int, default=15, help="runs per case")
    parser.add_argument("--out", default=str(ROOT / "runs/door_memory_check.json"))
    args = parser.parse_args()
    report = {}
    for case in ("case3", "case4", "case5"):
        tally = {
            "v1_diff": 0.0,
            "ms": [],
            "true_doors": 0,
            "remembered": 0,
            "confirmed": 0,
            "correct": 0,
            "error_m": [],
            "rows": 0,
            "rows_with_door": 0,
            "across_checked": 0,
            "across_right": 0,
            "across_no_true_door": 0,
        }
        folders = sorted(
            Path(n).parent for n in glob.glob(f"{args.data}/{case}/part*/*/trajectory.npz")
        )
        rng = np.random.default_rng(0)
        for folder in rng.choice(folders, min(args.runs, len(folders)), replace=False):
            check_run(Path(folder), tally)

        def share(a, b):
            return round(a / b, 3) if b else None

        report[case] = {
            "runs": min(args.runs, len(folders)),
            "v1_unchanged_max_diff": tally["v1_diff"],
            "remembered": share(tally["remembered"], tally["true_doors"]),
            "true_doors": tally["true_doors"],
            "correct": share(tally["correct"], tally["confirmed"]),
            "confirmed": tally["confirmed"],
            "error_m_median": round(float(np.median(tally["error_m"])), 3)
            if tally["error_m"]
            else None,
            "rows_with_door": share(tally["rows_with_door"], tally["rows"]),
            "across_right": share(tally["across_right"], tally["across_checked"]),
            "across_checked": tally["across_checked"],
            "across_no_true_door": tally["across_no_true_door"],
            "ms_per_scan": round(float(np.mean(tally["ms"])), 2),
        }
        print(case, json.dumps(report[case]), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
