"""Run the full-knowledge RRT expert and the visible-only one on the same scenarios, and table the results.

Clean = target reached with no contact. HALF is the room half-size for fit_to_room; 0 keeps the scenario's own room.

Example:
    python compare_experts.py --name visible-expert --set moving scenarios_obstacles_moving_val_a.json 5 \
        --set floorplan scenarios_floorplan_val.json 0 --workers 10
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
EXPERTS = {"full": "rrt", "visible": "rrt_visible"}


def _episode(job):
    expert, name, scenario, folder, duration = job
    from obstacle_env import rollout
    from pipeline import read_json
    config = read_json(ROOT/"settings.json")
    config["duration_s"] = duration
    summary = rollout(config, {**scenario, "expert": EXPERTS[expert]}, folder, save_every=10)
    z = np.load(Path(folder)/"trajectory.npz")
    return {"expert": expert, "set": name, "id": scenario["id"], "clean": bool(summary["success"]),
            "reached": bool(summary["reached_target"]), "contact_steps": int(summary["contact_steps"]),
            "mover_contact": bool(z["contact_obstacle"].any()) if "contact_obstacle" in z.files else False,
            "wall_contact": bool(z["contact_wall"].any()) if "contact_wall" in z.files else False,
            "arrival_s": summary.get("base_arrival_time_s"), "time_s": float(summary["simulation_time_s"])}


def table(rows):
    out = []
    for name in dict.fromkeys(r["set"] for r in rows):
        for expert in EXPERTS:
            part = [r for r in rows if r["set"] == name and r["expert"] == expert]
            if part:
                arrivals = [r["arrival_s"] for r in part if r["arrival_s"] is not None]
                out.append({"set": name, "expert": expert, "episodes": len(part), "clean": sum(r["clean"] for r in part),
                            "reached": sum(r["reached"] for r in part), "contact_runs": sum(r["contact_steps"] > 0 for r in part),
                            "mover_contact_runs": sum(r["mover_contact"] for r in part),
                            "wall_contact_runs": sum(r["wall_contact"] for r in part),
                            "mean_arrival_s": float(np.mean(arrivals)) if arrivals else None})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="results go to runs/NAME")
    parser.add_argument("--set", nargs=3, action="append", required=True, metavar=("NAME", "FILE", "HALF"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--duration", type=float, default=110.0)
    args = parser.parse_args()

    from obstacle_env import fit_to_room
    out = ROOT/"runs"/args.name
    jobs = []
    for name, file, half in args.set:
        for s in json.loads((ROOT/file).read_text()):
            assert s["split"] != "test", "The test split is reserved for the final evaluation"
            if float(half) > 0:
                try:
                    s = fit_to_room(s, float(half))[0]
                except ValueError as e:
                    print(f"skipped {e}")
                    continue
            jobs += [(expert, name, s, out/expert/name/s["id"], args.duration) for expert in EXPERTS]
    rows = []
    with Pool(args.workers) as pool:
        for row in pool.imap_unordered(_episode, jobs):
            rows.append(row)
            print(row["expert"], row["id"], "clean" if row["clean"] else f"FAILED contact {row['contact_steps']}", flush=True)
            (out/"results.json").write_text(json.dumps({"table": table(rows), "episodes": rows}, indent=1))
    for row in table(rows):
        print(row)


if __name__ == "__main__":
    main()
