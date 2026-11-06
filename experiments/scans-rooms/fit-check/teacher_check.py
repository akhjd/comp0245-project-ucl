"""Teacher comparison (3-4 Nov): which room teacher can a small network copy? Same 40 train + 20 validation rooms
for every teacher (runs/teacher-compare/ids.json), 36 rays, 3 scans, no velocity, 2 x 64 fit as oracle_check.py.
Per teacher: its own result on the 20 validation rooms, copy R2 on its validation rows (all / mover within 1 m),
how often its command jumps, and arrival time. Pick rule (AGENT_LOG 22:47): among teachers clean on at least 15
of 20, the highest copy R2 near movers; none clearly above the current RRT = the teacher is not the fix."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from why_no_learning import ROOT, NO_VEL, STRIDE
from oracle_check import fit
from models import base_inputs
from train_base_torch import stack_scans

IDS = json.loads((ROOT/"runs/teacher-compare/ids.json").read_text())
TEACHERS = {"rrt_visible (current)": ("runs/rooms-expert", "runs/rooms-expert-val"),
            "rrt_best": ("runs/teacher-compare/rrt_best", "runs/teacher-compare/rrt_best-val"),
            "window": ("runs/teacher-compare/window", "runs/teacher-compare/window-val")}


def rows(folder, ids):
    """Inputs (3 scans, no velocity), wheel labels, nearest-mover distance and per-run jump flags of the successful
    runs among ids. Also the outcome of every listed run."""
    folder = ROOT/folder
    episodes = {e["path"]: e for e in json.loads((folder/"manifest.json").read_text())["episodes"]}
    xs, ys, near, jumps, runs = [], [], [], [], []
    for i in ids:
        e = episodes[i]
        moving = np.array(e.get("obstacle_is_moving", []), bool)
        hits = np.array(e.get("contact_steps_per_obstacle", []), bool)
        runs.append({"success": e["success"], "mover_hit": bool((hits & moving).any()) if len(moving) else False,
                     "arrival_s": e.get("base_arrival_time_s")})
        if not e["success"]:
            continue
        z = np.load(folder/i/"trajectory.npz")
        obs, lidar, y = z["observations"][::STRIDE], z["lidar"][::STRIDE], z["actions"][::STRIDE, :4]
        base = np.array([base_inputs(r) for r in obs])
        xs.append(np.hstack([base, stack_scans(lidar, 3, 50)])[:, [k for k in range(45) if k in NO_VEL] + list(range(45, 45 + 72))])
        ys.append(y)
        jumps.append(np.abs(y[10:] - y[:-10]).max(axis=1))
        xy = z["obstacle_xy"][::STRIDE][:, moving] if moving.any() else None
        near.append(np.linalg.norm(xy - z["base_m"][::STRIDE][:, None, :2], axis=2).min(axis=1) if xy is not None else np.full(len(y), 99.0))
    return (np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32), np.concatenate(near),
            np.concatenate(jumps), runs)


if __name__ == "__main__":
    torch.set_num_threads(6)
    out = {}
    for name, (train, val) in TEACHERS.items():
        if not (ROOT/val/"manifest.json").exists() or (name in sys.argv[2:] if sys.argv[1:2] == ["--skip"] else False):
            print(f"{name}: not recorded yet, skipped", flush=True)
            continue
        x, y, _, _, train_runs = rows(train, IDS["train"])
        xv, yv, near, jump, val_runs = rows(val, IDS["validation"])
        arrivals = [r["arrival_s"] for r in val_runs if r["success"] and r["arrival_s"] is not None]
        out[name] = {"train_clean": f"{sum(r['success'] for r in train_runs)}/{len(train_runs)}",
                     "validation_clean": f"{sum(r['success'] for r in val_runs)}/{len(val_runs)}",
                     "validation_mover_hits": sum(r["mover_hit"] for r in val_runs),
                     "arrival_median_s": round(float(np.median(arrivals)), 1) if arrivals else None,
                     "command_jumps_pct": round(100*float((jump > y.std(0).mean()).mean()), 1),
                     "copy_r2": fit(x, y, xv, yv, near < 1.0)}
        print(name, out[name], flush=True)
        (HERE/"teacher_check.json").write_text(json.dumps(out, indent=1))
