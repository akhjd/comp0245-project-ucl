"""Cheat-input check on the dynamic window teacher (4 Nov). The window expert knows every recently seen mover's exact
position and velocity; the model only has 3 lidar scans 0.5 s apart. Does giving the model the movers' true state
make it copy the teacher much better near movers?

Same inputs as the window DAgger (3 scans, no velocity, 72 rays), plain cloning on runs/rooms-window72 (successful
runs), scored on rooms-window72-val; 2 x 64 network as oracle_check.py, 12 epochs, best validation R2, 2 seeds.
  normal  the model's own inputs
  movers  + true position and velocity of the 3 nearest movers, robot frame (as oracle_check.py)
Rule written before the results (AGENT_LOG 4 Nov): R2 near movers (mover within 1 m) up by >= 0.1 over normal (seed
mean) -> seeing mover motion is the bottleneck; otherwise it is room variety / margins, not inputs."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")    # runs/ is a symlink to /data, so not from __file__
HERE = ROOT/"runs/scans-rooms/fit-check"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from train_base_torch import load, input_names, dataset_rays, VELOCITY, STRIDE
from oracle_check import fit

OUT = HERE/"window_oracle_check.json"
PAD = [5.0, 0.0, 0.0, 0.0]      # no mover: "far ahead, still"
results = json.loads(OUT.read_text()) if OUT.exists() else {}


def say(name, value):
    results[name] = value
    print(name, value, flush=True)
    OUT.write_text(json.dumps(results, indent=1))


def movers(folder, split):
    """Per row of load(split, folder): the 3 nearest movers' (forward, left, forward speed, left speed) in the robot's
    frame, and the nearest mover's distance. Same episodes, rows and order as load."""
    manifest = json.loads((folder/"manifest.json").read_text())
    stride = max(1, STRIDE // manifest.get("saved_every", 1))
    info, near = [], []
    for ep in manifest["episodes"]:
        if ep["split"] != split or not ep["success"]:
            continue
        z = np.load(folder/ep["path"]/"trajectory.npz")
        obs = z["observations"][::stride]
        n = len(obs)
        rows = np.tile(np.array(PAD*3, np.float32), (n, 1))
        moving = np.array(ep.get("obstacle_is_moving", []), bool)
        if not moving.any():
            info.append(rows)
            near.append(np.full(n, 99.0))
            continue
        xy = z["obstacle_xy"][::stride][:, moving]
        speed = np.gradient(xy, 0.001*stride*manifest.get("saved_every", 1), axis=0)
        rel = xy - z["base_m"][::stride][:, None, :2]
        yaw = np.arctan2(obs[:, 3], obs[:, 4])[:, None]
        c, s = np.cos(yaw), np.sin(yaw)
        frame = np.stack([c*rel[..., 0] + s*rel[..., 1], c*rel[..., 1] - s*rel[..., 0],
                          c*speed[..., 0] + s*speed[..., 1], c*speed[..., 1] - s*speed[..., 0]], axis=2)
        dist = np.linalg.norm(rel, axis=2)
        k = np.argsort(dist, axis=1)[:, :3]
        picked = np.take_along_axis(frame, k[:, :, None], axis=1).reshape(n, -1)
        rows[:, :picked.shape[1]] = picked
        info.append(rows)
        near.append(dist.min(axis=1))
    return np.concatenate(info).astype(np.float32), np.concatenate(near)


if __name__ == "__main__":
    torch.set_num_threads(8)
    t = time.time()
    data, val = ROOT/"runs/rooms-window72", ROOT/"runs/rooms-window72-val"
    names = input_names(3, dataset_rays(data))
    columns = [i for i, n in enumerate(names) if n not in VELOCITY]
    x, y = load("train", data, scans=3)[:2]
    xv, yv = load("validation", val, scans=3)[:2]
    x, xv = x[:, columns], xv[:, columns]
    m, _ = movers(data, "train")
    mv, dv = movers(val, "validation")
    near = dv < 1.0
    say("rows", {"train": len(x), "validation": len(xv), "inputs": x.shape[1],
                 "validation_mover_within_1m_pct": round(100*float(near.mean()), 1), "load_s": round(time.time() - t)})
    for seed in (0, 1):
        if f"normal-seed{seed}" not in results:
            say(f"normal-seed{seed}", fit(x, y, xv, yv, near, seed=seed))
        if f"movers-seed{seed}" not in results:
            say(f"movers-seed{seed}", fit(np.hstack([x, m]), y, np.hstack([xv, mv]), yv, near, seed=seed))
    for kind in ("normal", "movers"):
        say(f"{kind}-mean", {k: round(float(np.mean([results[f"{kind}-seed{s}"][k] for s in (0, 1)])), 3)
                             for k in ("all", "mover_within_1m")})
    say("seconds", round(time.time() - t))
