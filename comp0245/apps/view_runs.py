"""Build a replay page (viewer.html) for a run folder: a top-down animation of every episode.

It shows the base path and heading, the hand, the target and the expert's parking spot, plus
end-effector error over time. With --expert, the expert's run on the same target is overlaid for
comparison.

Examples:
    python -m comp0245.apps.view_runs archive/runs/mlp-v3-validation --expert runs/dataset-v3
    python -m comp0245.apps.view_runs runs/dataset-v3 --split validation
Then open the printed viewer.html in a browser.
"""

import argparse
import json
from pathlib import Path

import numpy as np

TEMPLATE = (Path(__file__).parent / "templates" / "run_viewer.html").read_text()
STEP_S = 0.02  # replay sample period; the 1 kHz logs are thinned to 50 Hz


def parking_spot(target):
    """The expert's base goal for this target (same rule as comp0245/sim/robot_env.py), drawn
    for reference only."""
    standoff = np.interp(target[2], [0.1, 0.5, 1.1, 1.25], [0.80, 0.65, 0.65, 0.35])
    bearing = np.arctan2(target[1], target[0]) if np.linalg.norm(target[:2]) > 1e-4 else 0.0
    return (target[:2] - standoff * np.array([np.cos(bearing), np.sin(bearing)])).tolist()


def load_episode(folder):
    with np.load(folder / "trajectory.npz", allow_pickle=False) as z:
        t, obs, base, ee, err, act = (
            z["time_s"],
            z["observations"],
            z["base_m"],
            z["ee_m"],
            z["error_m"],
            z["actions"],
        )
        lidar = z["lidar"] if "lidar" in z.files else None
    keep = np.unique(np.searchsorted(t, np.arange(0, t[-1], STEP_S)))

    def r(a):
        return np.round(a[keep], 3).tolist()

    summary = json.loads((folder / "summary.json").read_text())
    target = obs[0, 5:8]
    scenario = (
        json.loads((folder / "episode.json").read_text())["scenario"]
        if (folder / "episode.json").exists()
        else {}
    )
    extra = {}
    if (
        "obstacles" in scenario
    ):  # obstacle runs: their own parking spot, obstacles and 10 Hz lidar (in cm)
        extra = {"park": scenario["park"], "obstacles": scenario["obstacles"]}
    if lidar is not None:
        extra["lidar"] = np.round(lidar[keep[::5]] * 100).astype(int).tolist()
    return {
        "t": r(t),
        "x": r(base[:, 0]),
        "y": r(base[:, 1]),
        "z": r(base[:, 2]),
        "yaw": r(np.arctan2(obs[:, 3], obs[:, 4])),
        "ex": r(ee[:, 0]),
        "ey": r(ee[:, 1]),
        "err": r(err),
        "wheels": np.round(act[keep, :4], 2).tolist(),
        "target": target.round(3).tolist(),
        "park": np.round(parking_spot(target), 3).tolist(),
        "success": summary["success"],
        "termination": summary["termination"],
        "final_error": round(summary["final_error_m"], 3),
        "controller": summary["controller"],
    } | extra


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", help="a dataset or evaluation folder containing episode subfolders")
    parser.add_argument(
        "--expert", help="dataset folder holding the expert runs on the same scenario IDs"
    )
    parser.add_argument("--split", help="only episodes whose ID starts with this, e.g. validation")
    args = parser.parse_args()
    run = Path(args.run)
    episodes = {}
    for folder in sorted(p for p in run.iterdir() if (p / "trajectory.npz").exists()):
        if args.split and not folder.name.startswith(args.split):
            continue
        episodes[folder.name] = {"main": load_episode(folder)}
        if args.expert and (Path(args.expert) / folder.name / "trajectory.npz").exists():
            episodes[folder.name]["expert"] = load_episode(Path(args.expert) / folder.name)
    if not episodes:
        raise ValueError(f"No episodes with trajectory.npz in {run}")
    page = TEMPLATE.replace("__TITLE__", run.name).replace("__DATA__", json.dumps(episodes))
    out = run / "viewer.html"
    out.write_text(page)
    print(f"Wrote {out.resolve()} ({len(episodes)} episodes)")


if __name__ == "__main__":
    main()
