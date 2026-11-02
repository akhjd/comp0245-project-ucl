"""Run the obstacle expert on a scenarios file and save one episode folder per scenario plus manifest.json.

Example:
    python collect_obstacles.py --scenarios scenarios_obstacles_v2.json --out runs/obst-v2b
    python collect_obstacles.py --scenarios scenarios_obstacles_v2.json --out runs/obst-gui --only train_000 --gui
"""
import argparse
import json
import shutil
from pathlib import Path
from models import FEATURE_NAMES, ACTION_NAMES, SCHEMA_VERSION
from pipeline import read_json, write_json, ROOT
from obstacle_env import rollout, fit_to_room, LIDAR_RAYS

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", nargs="*", help="scenario IDs to run")
    parser.add_argument("--duration", type=float, default=70.0, help="seconds per episode")
    parser.add_argument("--room", type=float, metavar="HALF",
                        help="square room with walls at +-HALF metres; scenarios that do not fit are skipped")
    parser.add_argument("--dot-order", choices=["best", "nearest"], default="best",
                        help="with several dots: compare every order, or always go to the closest next")
    parser.add_argument("--rrt", action="store_true", help="use the privileged RRT expert (RRTExpert): true obstacles, no lidar")
    parser.add_argument("--save-every", type=int, default=1, help="keep every Nth control step in trajectory.npz")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted collection: keep finished episodes, redo unfinished ones, and log a failed episode instead of stopping")
    parser.add_argument("--only-clean", action="store_true",
                        help="delete the trajectory of any run that did not succeed without contact; its summary stays, for the count")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    config = read_json(ROOT/"settings.json")
    config["duration_s"] = args.duration
    scenarios = json.loads(Path(args.scenarios).read_text())
    if args.only:
        scenarios = [s for s in scenarios if s["id"] in args.only]
    if args.room:
        fitted = []
        for s in scenarios:
            try:
                s, dropped = fit_to_room(s, args.room)
            except ValueError as e:
                print(f"skipped {e}")
                continue
            if dropped:
                print(f"{s['id']}: dropped {dropped} obstacle(s) outside the room")
            fitted.append(s)
        scenarios = fitted
    scenarios = [{**s, "dot_order": args.dot_order} for s in scenarios]
    if args.rrt:
        scenarios = [{**s, "expert": "rrt"} for s in scenarios]
    out = Path(args.out)
    if out.exists() and not args.resume:
        raise FileExistsError(f"Choose a new output folder: {out}")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": SCHEMA_VERSION, "feature_names": FEATURE_NAMES, "action_names": ACTION_NAMES,
                "lidar_rays": LIDAR_RAYS, "settings": config, "scenarios_file": str(args.scenarios),
                "scenarios": scenarios, "episodes": []}
    for s in scenarios:
        print(f"{s['id']} ({len(s['obstacles'])} obstacles) ...", flush=True)
        if args.resume and (out/s["id"]/"summary.json").exists():
            manifest["episodes"].append({"path": s["id"], **read_json(out/s["id"]/"summary.json")})
            continue
        if args.resume and (out/s["id"]).exists():
            shutil.rmtree(out/s["id"])
        try:
            summary = rollout(config, s, out/s["id"], gui=args.gui, save_every=args.save_every)
        except Exception as error:
            if not args.resume:
                raise
            print(f"  FAILED: {error}", flush=True)
            continue
        manifest["episodes"].append({"path": s["id"], **summary})
        write_json(out/"manifest.json", manifest)
        if args.only_clean and not summary["success"]:
            (out/s["id"]/"trajectory.npz").unlink()
        print(f"  {summary['termination']}: hand error {summary['final_error_m']:.4f} m, "
              f"standoff error {summary['final_standoff_error_m']:.3f} m, contacts {summary['contact_steps']}, "
              f"closest lidar {summary['min_lidar_m']:.2f} m, clearance {summary['min_clearance_m']:.2f} m, arrived at {summary['base_arrival_time_s']}", flush=True)
    good = sum(e["success"] for e in manifest["episodes"])
    print(f"{good}/{len(manifest['episodes'])} successful (target reached, no contact)")

if __name__ == "__main__":
    main()
