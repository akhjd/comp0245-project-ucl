"""Re-score saved base evaluations with the current eval_obstacles.score, without running the simulator.

Written after Codex's 28 Oct review: the old strict/loose checks looked at one saved row and ignored turning.
The current scorer needs the parking to hold for the last 2 s with the base stopped (speed and yaw rate).
The original results are left untouched; the new scores go next to them as rescored.json. One limit: runs saved
before the fix never checked contact after the final simulator step (at most one step of 40,000 is missed).

Handles robustness runs (runs/robust-*/results.json) and evaluation folders with an evaluation.json
(DAgger rK-validation folders, eval_obstacles.py outputs).

Example:
    python rescore.py runs/robust-base-r2 runs/robust-noise-r2 runs/dagger-*/r*-validation
"""
import argparse
import json
from pathlib import Path
from eval_obstacles import score
from pipeline import ROOT
from robustness import summarise

# Episode ids are unique within each set; robustness runs use the fresh and hard validation sets, the DAgger and
# eval_obstacles validation folders the 10 reference scenarios of scenarios_obstacles_v2.json
ROBUST_SCENARIOS = ["scenarios_obstacles_val100.json", "scenarios_obstacles_hard.json"]
REFERENCE_SCENARIOS = ["scenarios_obstacles_v2.json"]


def scenarios(files):
    return {s["id"]: s for f in files for s in json.loads((ROOT/f).read_text())}


def rescore_episode(folder, scenario):
    return score(folder, scenario, json.loads((folder/"summary.json").read_text()))


def rescore_robustness(root):
    old = json.loads((root/"results.json").read_text())
    by_id = scenarios(ROBUST_SCENARIOS)
    keys, scores = [], []
    for e in old["episodes"]:
        key = (e["controller"], e["test"], e["level"])
        keys.append(key)
        scores.append(rescore_episode(root/e["controller"]/f"{e['test']}-{e['level']}"/e["id"], by_id[e["id"]]))
    print(f"== {root}")
    table = summarise(keys, scores)
    (root/"rescored.json").write_text(json.dumps({"scorer": "eval_obstacles.score, 2 s settle window",
        "controllers": old["controllers"], "tests": old["tests"], "table": table,
        "episodes": [dict(zip(("controller", "test", "level"), k), **r) for k, r in zip(keys, scores)]}, indent=1))


def rescore_evaluation(folder):
    old = json.loads((folder/"evaluation.json").read_text())
    by_id = scenarios(REFERENCE_SCENARIOS)
    scores = [rescore_episode(folder/e["id"], by_id[e["id"]]) for e in old["episodes"]]
    n = len(scores)
    totals = {"episodes": n, "strict": sum(r["strict"] for r in scores), "loose": sum(r["loose"] for r in scores),
              "no_contact": sum(r["contact_steps"] == 0 for r in scores),
              "reachable": sum(r["reachable"] for r in scores)}
    before = old["totals"]
    print(f"{str(folder):45s} strict {before['strict']:2d} -> {totals['strict']:2d}, "
          f"loose {before['loose']:2d} -> {totals['loose']:2d}  (of {n})")
    (folder/"rescored.json").write_text(json.dumps({"scorer": "eval_obstacles.score, 2 s settle window",
        "totals_before": before, "totals": totals, "episodes": scores}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+")
    for folder in map(Path, parser.parse_args().folders):
        if (folder/"results.json").exists():
            rescore_robustness(folder)
        elif (folder/"evaluation.json").exists():
            rescore_evaluation(folder)
        else:
            print(f"skipped {folder}: no results.json or evaluation.json")


if __name__ == "__main__":
    main()
