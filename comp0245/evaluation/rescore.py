"""Re-score saved base evaluations with the current eval_obstacles.score, without running the
simulator.

Written after Codex's 28 Oct review: the old strict/loose checks looked at one saved row and
ignored turning. The current scorer needs the parking to hold for the last 2 s with the base
stopped (speed and yaw rate). The original results are left untouched; the new scores go next to
them as rescored.json. One limit: runs saved before the fix never checked contact after the final
simulator step (at most one step of 40,000 is missed).

Handles robustness runs (runs/robust-*/results.json), evaluation folders with an evaluation.json
(DAgger rK-validation folders, comp0245/evaluation/eval_obstacles.py outputs) and expert datasets
(manifest.json only, e.g. runs/rooms-expert-val).

3 Nov: also adds the "parked" score (eval_obstacles.score; reach area, facing, still, no contact),
now the reported base metric. A folder that already has a rescored.json (from 28 Oct) gets
rescored-parked.json instead, so earlier rescorings stay as they were. Each episode's scenario is
read from its own episode.json when there is one.

Example:
    python -m comp0245.evaluation.rescore runs/robust-base-r2 runs/robust-noise-r2 \\
        runs/dagger-*/r*-validation
    python -m comp0245.evaluation.rescore runs/dagger-rooms-*/r*-validation runs/rooms-expert-val
"""

import argparse
import json
from pathlib import Path

from comp0245.evaluation.eval_obstacles import score
from comp0245.evaluation.robustness import summarise
from comp0245.paths import scenario_path

# Episode ids are unique within each set; robustness runs use the fresh and hard validation sets,
# the DAgger and
# eval_obstacles validation folders the 10 reference scenarios of scenarios_obstacles_v2.json
ROBUST_SCENARIOS = ["scenarios_obstacles_val100.json", "scenarios_obstacles_hard.json"]
REFERENCE_SCENARIOS = ["scenarios_obstacles_v2.json"]


def scenarios(files):
    return {s["id"]: s for f in files for s in json.loads(scenario_path(f).read_text())}


def rescore_episode(folder, scenario=None):
    if (folder / "episode.json").exists():  # the scenario exactly as it was run
        scenario = json.loads((folder / "episode.json").read_text())["scenario"]
    return score(folder, scenario, json.loads((folder / "summary.json").read_text()))


OUT_NAME = (
    None  # --as: write this file name instead (e.g. rescored-reach.json for the 4 Nov reach rule)
)


def output(folder):
    """rescored.json, or rescored-parked.json when an earlier rescoring is already there (never
    overwritten)."""
    name = OUT_NAME or (
        "rescored-parked.json" if (folder / "rescored.json").exists() else "rescored.json"
    )
    if (folder / name).exists():
        raise FileExistsError(f"{folder / name} exists already")
    return folder / name


def totals_of(scores):
    return {
        "episodes": len(scores),
        "parked": sum(r["parked"] for r in scores),
        "parked_worst_case_reach": sum(
            r.get("parked_worst_case_reach", r["parked"]) for r in scores
        ),
        "strict": sum(r["strict"] for r in scores),
        "loose": sum(r["loose"] for r in scores),
        "no_contact": sum(r["contact_steps"] == 0 for r in scores),
        "reachable": sum(r["reachable"] for r in scores),
    }


SCORER = (
    "eval_obstacles.score: parked (reach area, facing +-30 deg, still 2 s, no contact); "
    "strict/loose for the record"
)


def rescore_robustness(root):
    old = json.loads((root / "results.json").read_text())
    by_id = scenarios(ROBUST_SCENARIOS)
    keys, scores = [], []
    for e in old["episodes"]:
        key = (e["controller"], e["test"], e["level"])
        keys.append(key)
        scores.append(
            rescore_episode(
                root / e["controller"] / f"{e['test']}-{e['level']}" / e["id"], by_id[e["id"]]
            )
        )
    print(f"== {root}")
    table = summarise(keys, scores)
    for row, key in zip(table, dict.fromkeys(keys), strict=False):
        row["parked"] = sum(r["parked"] for k, r in zip(keys, scores, strict=False) if k == key)
        print(f"{key[0]:11s} {key[1]:6s} level {key[2]}: parked {row['parked']}/{row['episodes']}")
    output(root).write_text(
        json.dumps(
            {
                "scorer": SCORER,
                "controllers": old["controllers"],
                "tests": old["tests"],
                "table": table,
                "episodes": [
                    dict(zip(("controller", "test", "level"), k, strict=False), **r)
                    for k, r in zip(keys, scores, strict=False)
                ],
            },
            indent=1,
        )
    )


def rescore_evaluation(folder):
    old = json.loads((folder / "evaluation.json").read_text())
    by_id = None
    scores = []
    for e in old["episodes"]:
        if not (folder / e["id"] / "episode.json").exists() and by_id is None:
            by_id = scenarios(REFERENCE_SCENARIOS)
        scores.append(rescore_episode(folder / e["id"], by_id[e["id"]] if by_id else None))
    totals = totals_of(scores)
    before = old["totals"]
    print(
        f"{str(folder):50s} parked {totals['parked']:2d} (83% rule "
        f"{totals['parked_worst_case_reach']:2d})  "
        f"(strict {before['strict']:2d} -> {totals['strict']:2d}, "
        f"loose {before['loose']:2d} -> {totals['loose']:2d}, of {totals['episodes']})"
    )
    output(folder).write_text(
        json.dumps(
            {"scorer": SCORER, "totals_before": before, "totals": totals, "episodes": scores},
            indent=1,
        )
    )


def rescore_dataset(folder):
    """Expert runs (no evaluation.json): the expert's base parking scored with the same rule as
    the models'."""
    episodes = json.loads((folder / "manifest.json").read_text())["episodes"]
    scores = [rescore_episode(folder / e["path"]) for e in episodes]
    totals = totals_of(scores)
    print(
        f"{str(folder):50s} parked {totals['parked']:2d}  (strict {totals['strict']:2d}, loose "
        f"{totals['loose']:2d}, "
        f"of {totals['episodes']}; expert task success {sum(e['success'] for e in episodes)})"
    )
    output(folder).write_text(
        json.dumps({"scorer": SCORER, "totals": totals, "episodes": scores}, indent=1)
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("folders", nargs="+")
    parser.add_argument(
        "--as",
        dest="as_name",
        help="output file name (default rescored.json / rescored-parked.json)",
    )
    args = parser.parse_args()
    global OUT_NAME
    OUT_NAME = args.as_name
    for folder in map(Path, args.folders):
        if (folder / "results.json").exists():
            rescore_robustness(folder)
        elif (folder / "evaluation.json").exists():
            rescore_evaluation(folder)
        elif (folder / "manifest.json").exists():
            rescore_dataset(folder)
        else:
            print(f"skipped {folder}: no results.json, evaluation.json or manifest.json")


if __name__ == "__main__":
    main()
