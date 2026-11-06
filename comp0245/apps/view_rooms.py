"""Build one page comparing several models on the same room scenarios: an outcome grid (scenario x
model) and top-down replays of every model on the chosen scenario, side by side on one clock, with
walls and moving obstacles.

Each model is NAME=FOLDER, a folder of evaluated room runs (evaluation.json + manifest.json, as
comp0245/training/dagger.py's rK-validation) or of expert runs (manifest.json only). With no models
given, the 3 Nov room comparison is used.

Example:
    python -m comp0245.apps.view_rooms --out runs/rooms-compare.html \\
        "Expert=runs/rooms-expert-val" \
        "Filter r3=runs/dagger-rooms-conv/r3-validation"
Then open the printed page in a browser.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from comp0245.evaluation.eval_obstacles import FACING_LIMIT, REACH_MIN_M, reach_limit
from comp0245.evaluation.rescore import rescore_episode
from comp0245.paths import ROOT

TEMPLATE = (Path(__file__).parent / "templates" / "room_compare.html").read_text()
STEP_S = 0.2  # replay sample period; the page interpolates between samples
HIT = {"none": 0, "mover": 1, "static": 2, "wall": 3}


def default_models():
    models = [
        ("Expert (visible RRT)", "runs/rooms-expert-val"),
        ("Final base (open floor)", "runs/scans-rooms/base-final-rerun/validation"),
        ("3 scans, DAgger r3", "runs/dagger-rooms-3scans/r3-validation"),
        ("Filter, DAgger r3", "runs/dagger-rooms-conv/r3-validation"),
        ("Threat, DAgger r3", "runs/dagger-rooms-threat/r3-validation"),
        ("Filter + threat, DAgger r3", "runs/dagger-rooms-conv-threat/r3-validation"),
    ]
    for label, run in (("Mixture", "dagger-rooms-mix"), ("Steady expert", "dagger-rooms-steady")):
        done = sorted(
            (ROOT / "runs" / run).glob("r*-validation/evaluation.json")
        )  # latest finished round
        if done:
            models.append(
                (
                    f"{label}, DAgger {done[-1].parent.name.split('-')[0]}",
                    str(done[-1].parent.relative_to(ROOT)),
                )
            )
    return models


def load_run(folder, ep, moving):
    """One episode thinned to STEP_S: base pose in cm / centiradians, obstacle centres in cm,
    and per sample the worst contact in that stretch (HIT codes), so a brief touch between
    samples is not lost."""
    with np.load(folder / ep / "trajectory.npz", allow_pickle=False) as z:
        t, base, obs = z["time_s"], z["base_m"], z["observations"]
        xy = z["obstacle_xy"] if "obstacle_xy" in z.files else None
        hit_obst = (
            z["contact_obstacle"] if "contact_obstacle" in z.files else np.zeros((len(t), 0), bool)
        )
        hit_wall = z["contact_wall"] if "contact_wall" in z.files else np.zeros(len(t), bool)
        contact = z["contact"]
    keep = np.unique(np.searchsorted(t, np.arange(0, t[-1], STEP_S)))
    code = np.zeros(len(t), int)
    code[contact] = HIT["static"]  # touching something: static unless shown otherwise below
    code[hit_wall] = HIT["wall"]
    if hit_obst.shape[1]:
        code[(hit_obst & ~np.asarray(moving, bool)).any(axis=1)] = HIT["static"]
        code[(hit_obst & np.asarray(moving, bool)).any(axis=1)] = HIT[
            "mover"
        ]  # a mover hit matters most
    worst = np.zeros(len(keep), int)
    for k, (a, b) in enumerate(zip(keep, list(keep[1:]) + [len(t)], strict=False)):
        seg = code[a:b]
        worst[k] = HIT["mover"] if (seg == HIT["mover"]).any() else seg.max(initial=0)
    yaw = np.arctan2(obs[keep, 3], obs[keep, 4])
    run = {
        "x": np.round(base[keep, 0] * 100).astype(int).tolist(),
        "y": np.round(base[keep, 1] * 100).astype(int).tolist(),
        "yaw": np.round(yaw * 100).astype(int).tolist(),
        "hit": worst.tolist(),
    }
    if xy is not None:
        run["obst"] = np.round(xy[keep] * 100).astype(int).reshape(len(keep), -1).tolist()
    return run


def load_model(name, folder):
    folder = ROOT / folder
    manifest = json.loads((folder / "manifest.json").read_text())
    expert = not (folder / "evaluation.json").exists()
    out = {}
    for e in manifest["episodes"]:
        ep = e["path"]
        moving = np.array(e.get("obstacle_is_moving", []), bool)
        hits = np.array(e.get("contact_steps_per_obstacle", []), bool)
        result = {
            "mover": bool((hits & moving).any()) if len(moving) else False,
            "static": bool((hits & ~moving).any()) if len(moving) else False,
            "wall": e.get("wall_contact_steps", 0) > 0,
        }
        # every column, the expert's too, scored with the same base rule (eval_obstacles.score
        # "parked")
        s = rescore_episode(folder / ep)
        why = (
            ""
            if s["parked"]
            else "touched something"
            if s["contact_steps"]
            else f"stopped {s['range_m']:.2f} m away, outside the reach area"
            if not s["in_reach_area"]
            else "not looking at the target"
            if s["worst_facing_error_rad"] > FACING_LIMIT
            else "did not stay still"
        )
        result |= {
            "park": "parked" if s["parked"] else "no",
            "why": why,
            "range_m": round(s["range_m"], 2),
            "path_m": round(s["base_path_length_m"], 1),
            "expert": expert,
            "task": bool(e["success"]) if expert else None,
        }
        out[ep] = {"result": result, "run": load_run(folder, ep, moving)}
    print(f"{name}: {len(out)} episodes from {folder.relative_to(ROOT)}")
    return out


def scenario_of(folder, ep):
    s = json.loads((ROOT / folder / ep / "episode.json").read_text())["scenario"]
    return {
        "room": s.get("room"),
        "target": [round(v, 2) for v in s["target"][:2]],
        "park": [round(v, 2) for v in s["park"]],
        "reach": [
            REACH_MIN_M,
            round(reach_limit(s["target"][2]), 3),
        ],  # accepted parking distances round the target
        "start": [round(v, 2) for v in s["start"]],
        "obstacles": [
            {"r": round(o["r"], 3), "moving": bool(np.any(o.get("vel", [0, 0])))}
            for o in s["obstacles"]
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("models", nargs="*", help="NAME=FOLDER, in display order")
    parser.add_argument("--out", default="runs/rooms-compare.html")
    args = parser.parse_args()
    models = [tuple(m.split("=", 1)) for m in args.models] or default_models()
    loaded = [load_model(name, folder) for name, folder in models]
    ids = sorted(set().union(*loaded))
    scenarios = {}
    for ep in ids:
        for (_name, folder), runs in zip(models, loaded, strict=False):
            if ep in runs:
                scenarios[ep] = scenario_of(folder, ep)
                break
    data = {
        "models": [name for name, _ in models],
        "folders": [folder for _, folder in models],
        "step": STEP_S,
        "scenarios": scenarios,
        "runs": {ep: [runs.get(ep) for runs in loaded] for ep in ids},
    }
    out = ROOT / args.out
    out.write_text(TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(
        f"Wrote {out} ({len(ids)} scenarios, {len(models)} models, {out.stat().st_size / 1e6:.1f} "
        "MB)"
    )


if __name__ == "__main__":
    main()
