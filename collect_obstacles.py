"""Run the obstacle expert on a scenarios file and save one episode folder per scenario plus manifest.json.

Example:
    python collect_obstacles.py --scenarios scenarios_obstacles_v2.json --out runs/obst-v2b
    python collect_obstacles.py --scenarios scenarios_obstacles_v2.json --out runs/obst-gui --only train_000 --gui
    python collect_obstacles.py --scenarios scenarios_rooms_train.json --out runs/rooms-72 --rrt --lidar-rays 72
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
from models import FEATURE_NAMES, ACTION_NAMES, SCHEMA_VERSION
from pipeline import read_json, write_json, ROOT
from obstacle_env import rollout, fit_to_room, LIDAR_RAYS

def run_episode(job):
    config, scenario, folder, gui = job
    return rollout(config, scenario, folder, gui=gui)

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
    parser.add_argument("--rrt", action="store_true", help="use the RRT expert, with true values only for movers the lidar hit (VisibleRRTExpert)")
    parser.add_argument("--full-knowledge", action="store_true", help="with --rrt: RRTExpert, which knows every obstacle and turning point")
    parser.add_argument("--steady", action="store_true", help="with --rrt: SteadyRRTExpert, which settles near-ties by side")
    parser.add_argument("--teacher", choices=["rrt_visible", "rrt_steady", "rrt_best", "window"],
                        help="base expert by name (overrides --rrt/--steady): rrt_best = BestRRTExpert, window = WindowExpert")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="episodes run at once")
    parser.add_argument("--lidar-rays", type=int, default=LIDAR_RAYS, help="lidar rays, evenly spaced (default 36)")
    parser.add_argument("--lidar-range", type=float, default=4.0, help="lidar range, m from the base centre (default 4)")
    args = parser.parse_args()
    config = read_json(ROOT/"settings.json")
    config["duration_s"] = args.duration
    config["lidar_rays"] = args.lidar_rays
    config["lidar_range"] = args.lidar_range
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
    if args.teacher:
        scenarios = [{**s, "expert": args.teacher} for s in scenarios]
    elif args.rrt:
        name = "rrt" if args.full_knowledge else "rrt_steady" if args.steady else "rrt_visible"
        scenarios = [{**s, "expert": name} for s in scenarios]
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Choose a new output folder: {out}")
    out.mkdir(parents=True)
    manifest = {"schema_version": SCHEMA_VERSION, "feature_names": FEATURE_NAMES, "action_names": ACTION_NAMES,
                "lidar_rays": args.lidar_rays, "lidar_range": args.lidar_range, "settings": config, "scenarios_file": str(args.scenarios),
                "scenarios": scenarios, "episodes": []}
    with Pool(args.workers) as pool:
        jobs = [(config, s, out/s["id"], args.gui) for s in scenarios]
        for s, summary in zip(scenarios, pool.imap(run_episode, jobs)):
            print(f"{s['id']} ({len(s['obstacles'])} obstacles)", flush=True)
            manifest["episodes"].append({"path": s["id"], **summary})
            write_json(out/"manifest.json", manifest)
            print(f"  {summary['termination']}: hand error {summary['final_error_m']:.4f} m, "
                  f"standoff error {summary['final_standoff_error_m']:.3f} m, contacts {summary['contact_steps']}, "
                  f"closest lidar {summary['min_lidar_m']:.2f} m, clearance {summary['min_clearance_m']:.2f} m, arrived at {summary['base_arrival_time_s']}", flush=True)
    good = sum(e["success"] for e in manifest["episodes"])
    print(f"{good}/{len(manifest['episodes'])} successful (target reached, no contact)")

if __name__ == "__main__":
    main()
