"""Rigorous driving test of the wheels MLP on many scenarios it has never seen, with confidence
ranges.

Driving is deterministic (the same scenario gives the same run), so repeats tell nothing new: the
test uses many scenarios. Fresh scenarios are drawn exactly like the dataset's (make_dataset.make)
with a different seed, so none was trained on or validated on. The test split can be added. With
--expert the RRT expert drives the same scenarios, to separate "the MLP failed" from "this scenario
is hard for anyone".

    python -m comp0245.evaluation.eval_mlp --out runs/eval1 --fresh 40                          \\
        40 fresh scenarios per case (200 runs, about 1 h)
    python -m comp0245.evaluation.eval_mlp --out runs/eval1 --fresh 40 --expert                 \\
        and the expert on the same ones (about 2-3 h more)
    python -m comp0245.evaluation.eval_mlp --out runs/eval1 --fresh 40 --test                   \\
        also the test split (keep it for the final result)
    python -m comp0245.evaluation.eval_mlp --out runs/eval1 --fresh 40 --report                 \\
        print the tables again from saved runs

Each run is saved in --out/<who>/<scenario id>/, so it can be stopped and run again to continue.
Results: --out/report.json. Groups: by case; still cylinders only vs with a mover; same room vs
needing a doorway (no straight route from start to the first dot's parking spot). Every rate has
a 95% Wilson range: with 15 runs, 13/15 means anywhere from 62% to 96%.
"""

import argparse
import collections
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from comp0245.paths import SETTINGS_FILE
from comp0245.training.dagger_mlp import CASES, drive, scenarios_of

CASE_NAMES = {
    "case1": "open room, no obstacles, 1 dot",
    "case2": "open room, cylinders, 1 dot",
    "case3": "floor plan, no cylinders, 1 dot",
    "case4": "floor plan, cylinders, 1 dot",
    "case5": "floor plan, cylinders, 3 dots",
}


def fresh(n, seed, cases):
    from comp0245.scenarios import make_dataset

    out = []
    for offset, case in enumerate(CASES):
        rng = np.random.default_rng([seed, offset])
        for i in range(n):
            s = make_dataset.make(case, rng)
            if case in cases:
                out.append(
                    {
                        **s,
                        "id": f"fresh{seed}_{case}_{i:03d}",
                        "split": "fresh",
                        "case": case,
                        "seed": int(rng.integers(1_000_000)),
                        "expert": "rrt",
                    }
                )
    return out


def tags(s):
    from comp0245.sim.obstacle_env import FloorPlanRoute

    moving = any(np.any(np.asarray(o.get("vel", [0, 0])) != 0) for o in s["obstacles"])
    t = {
        "case": s["case"],
        "cylinders": "with a mover" if moving else "still only" if s["obstacles"] else "none",
    }
    if s.get("boxes"):
        lo = np.array([b["min"] for b in s["boxes"]], float)
        hi = np.array([b["max"] for b in s["boxes"]], float)
        route = FloorPlanRoute(s["room"], lo, hi, [], [])
        t["route"] = (
            "same room"
            if route.sees(np.array(s["start"][:2]), np.array(s["park"]))
            else "needs a doorway"
        )
    return t


def drive_expert(job):
    """The RRT expert on one scenario, in its own process (as
    comp0245/training/collect_obstacles.py --rrt)."""
    _, scenario, folder, duration = job
    import shutil

    from comp0245.pipeline import read_json
    from comp0245.sim.obstacle_env import rollout

    summary_file = Path(folder) / "summary.json"
    if summary_file.exists():
        return json.loads(summary_file.read_text())
    shutil.rmtree(folder, ignore_errors=True)
    config = read_json(SETTINGS_FILE)
    config["duration_s"] = duration
    return rollout(config, scenario, folder, save_every=10)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    mid, half = (
        (p + z * z / (2 * n)) / (1 + z * z / n),
        z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n),
    )
    return max(0.0, mid - half), min(1.0, mid + half)


def contact_kind(r, s):
    if not r["contact_steps"]:
        return None
    per = r.get("contact_steps_per_obstacle", [])
    moving = [np.any(np.asarray(o.get("vel", [0, 0])) != 0) for o in s["obstacles"]]
    if any(n and m for n, m in zip(per, moving, strict=False)):
        return "mover"
    if any(n and not m for n, m in zip(per, moving, strict=False)):
        return "still cylinder"
    return "wall or furniture"


def table(results, scenarios, who):
    by = collections.defaultdict(list)
    for r, s in zip(results, scenarios, strict=False):
        t = tags(s)
        by[("all", "")].append((r, s))
        by[("case", f"{t['case']}: {CASE_NAMES[t['case']]}")].append((r, s))
        by[("cylinders", t["cylinders"])].append((r, s))
        if "route" in t:
            by[("floor plans", t["route"])].append((r, s))
            by[("floor plans, cylinders", f"{t['route']}, {t['cylinders']}")].append((r, s))
    rows = {}
    print(f"\n{who}: {len(results)} runs")
    print(
        f"  {'group':58s} {'clean':>9s}  {'95% range':>11s}  {'dots':>9s}  how runs ended / what "
        "was touched"
    )
    for (kind, name), group in sorted(
        by.items(),
        key=lambda kv: (
            ["all", "case", "cylinders", "floor plans", "floor plans, cylinders"].index(kv[0][0]),
            kv[0][1],
        ),
    ):
        k, n = sum(r["success"] for r, _ in group), len(group)
        lo, hi = wilson(k, n)
        dots = sum(r["dots_picked"] for r, _ in group), sum(r["dots_total"] for r, _ in group)
        ended = dict(collections.Counter(r["termination"] for r, _ in group))
        touched = dict(collections.Counter(c for c in (contact_kind(r, s) for r, s in group) if c))
        label = f"{kind}: {name}" if name else kind
        print(
            f"  {label:58s} {k:3d}/{n:<3d} {100 * k / n:3.0f}%  {100 * lo:3.0f}-{100 * hi:3.0f}%  "
            f"{dots[0]:4d}/{dots[1]:<4d}  {ended} {touched or ''}"
        )
        rows[label] = {
            "clean": k,
            "runs": n,
            "rate": k / n,
            "range95": [lo, hi],
            "dots": dots,
            "ended": ended,
            "touched": touched,
        }
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="runs/dagger-pre/r3/best.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fresh", type=int, default=40, help="fresh scenarios per case")
    parser.add_argument(
        "--seed", type=int, default=7, help="seed for the fresh scenarios (the dataset used 2026)"
    )
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=CASES)
    parser.add_argument("--test", action="store_true", help="also drive the test split")
    parser.add_argument(
        "--expert", action="store_true", help="also let the RRT expert drive the same scenarios"
    )
    parser.add_argument(
        "--report", action="store_true", help="only print the tables from saved runs"
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--data", default="runs/dataset")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    listed = out / "scenarios.json"
    if listed.exists():
        scenarios = json.loads(listed.read_text())
    else:
        scenarios = fresh(args.fresh, args.seed, args.cases)
        if args.test:
            found = scenarios_of(args.data, "test")
            scenarios += [s for c in args.cases for s in found[c]]
        listed.write_text(json.dumps(scenarios))
    print(
        f"{len(scenarios)} scenarios: "
        + ", ".join(f"{c} {sum(s['case'] == c for s in scenarios)}" for c in CASES),
        flush=True,
    )
    report = {"model": args.model, "scenarios": len(scenarios)}
    for who, job in (("mlp", drive), ("expert", drive_expert)):
        if who == "expert" and not args.expert:
            continue
        folder = out / who
        jobs = [(args.model, s, str(folder / s["id"]), args.duration) for s in scenarios]
        if args.report:
            done = [
                (s, json.loads((folder / s["id"] / "summary.json").read_text()))
                for s in scenarios
                if (folder / s["id"] / "summary.json").exists()
            ]
            ran, results = [s for s, _ in done], [r for _, r in done]
        else:
            begin = time.time()
            with mp.get_context("spawn").Pool(args.workers) as pool:
                results = pool.map(job, jobs, chunksize=1)
            ran = scenarios
            print(
                f"{who}: {len(results)} runs in {(time.time() - begin) / 60:.0f} min", flush=True
            )
        if results:
            report[who] = table(results, ran, who)
    if "mlp" in report and "expert" in report:
        mlp = {
            s["id"]: json.loads((out / "mlp" / s["id"] / "summary.json").read_text())["success"]
            for s in scenarios
            if (out / "mlp" / s["id"] / "summary.json").exists()
        }
        exp = {
            s["id"]: json.loads((out / "expert" / s["id"] / "summary.json").read_text())["success"]
            for s in scenarios
            if (out / "expert" / s["id"] / "summary.json").exists()
        }
        both = [i for i in mlp if i in exp]
        solved = [i for i in both if exp[i]]
        print(
            f"\nOn the {len(solved)} scenarios the expert solved cleanly, the MLP solved "
            f"{sum(mlp[i] for i in solved)} "
            f"({100 * sum(mlp[i] for i in solved) / max(len(solved), 1):.0f}%). The expert failed "
            f"{len(both) - len(solved)} of {len(both)}."
        )
        report["mlp_on_expert_solved"] = {
            "solved_by_expert": len(solved),
            "mlp_clean": sum(mlp[i] for i in solved),
        }
    (out / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
