"""Plan step 3: preprocess every saved run's lidar into the MLP's features (preprocess.py) and cache them next to the run as
features_<version>.npy. Training (train_mlp.py --preprocess) reads the cache, and computes any missing file itself, so this
script only does the work in parallel up front. Runs already cached are skipped, so it can be stopped and started again.

    python build_features.py --workers 6                                  all expert runs in runs/dataset
    python build_features.py --folders runs/dagger-pre/r1-data --workers 6    DAgger runs too
    python build_features.py --version v2 --workers 6                     with the remembered doorways (--door-memory)
"""
import argparse
import functools
import glob
import json
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np


def build(folder, version="v1"):
    import preprocess
    from train_mlp import MAX_DOTS, state_rows
    folder = Path(folder)
    if (folder/f"features_{version}.npy").exists():
        return 0
    z = np.load(folder/"trajectory.npz")
    summary = json.loads((folder/"summary.json").read_text())
    scenario = json.loads((folder/"episode.json").read_text())["scenario"]
    obs, t = z["observations"], z["time_s"]
    targets = np.array(scenario.get("targets", [scenario["target"]]))
    rows, _ = state_rows(obs, t, targets, {p["dot"]: p["time_s"] for p in summary["picks"]})
    preprocess.cached_features(folder, t, z["lidar"], obs, rows[:, 7:].reshape(-1, MAX_DOTS, 4), version)
    return len(t)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="runs/dataset", help="expert dataset (case*/part*/<run>/)")
    parser.add_argument("--folders", nargs="*", default=[], help="extra folders of runs (<folder>/<run>/), such as DAgger data")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--version", default="v1", choices=["v1", "v2"], help="feature set: v1, or v2 with the remembered doorways")
    args = parser.parse_args()
    runs = sorted(str(Path(n).parent) for n in glob.glob(f"{args.data}/case*/part*/*/trajectory.npz"))
    for extra in args.folders:
        runs += sorted(str(Path(n).parent) for n in glob.glob(f"{extra}/*/trajectory.npz"))
    begin, rows = time.time(), 0
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for k, n in enumerate(pool.imap_unordered(functools.partial(build, version=args.version), runs, chunksize=4), start=1):
            rows += n
            if k % 100 == 0 or k == len(runs):
                print(f"{k}/{len(runs)} runs, {rows} new rows, {(time.time() - begin)/60:.1f} min", flush=True)
    print("Done. Cached features are reused by train_mlp.py --preprocess.")


if __name__ == "__main__":
    main()
