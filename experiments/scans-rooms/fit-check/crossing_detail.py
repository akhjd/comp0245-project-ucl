"""Crossing mover hits of window r3 in detail (4 Nov). For each hit where a mover ran into the robot: the robot's speed
and turning over the 4 s before, and what would have happened from 2 s before contact if the robot had kept its
velocity, stopped, driven forward at full speed (0.25 m/s) or reversed (-0.1 m/s), each straight along its heading,
against the mover's ACTUAL recorded path. Footprint: wheel-box rectangle (x 0.342, y 0.313) + arm discs."""
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path("/home/akhjd/ros2_ws/comp0245_project/runs/scans-rooms/fit-check")))
from fail_types import scenario

DT, HX, HY = 0.01, 0.342, 0.313
ARM, ARM_R = np.array([0.46, 0.56, 0.66]), 0.07


def gap(p, yaw, c, r):
    """Clearance between the robot footprint (rectangle + arm discs) at p, yaw and a cylinder at c, radius r."""
    h = np.stack([np.cos(yaw), np.sin(yaw)], -1)
    n = np.stack([-h[..., 1], h[..., 0]], -1)
    d = c - p
    lx, ly = np.abs((d*h).sum(-1)), np.abs((d*n).sum(-1))
    rect = np.hypot(np.maximum(lx - HX, 0), np.maximum(ly - HY, 0)) - r
    arm = np.min([np.linalg.norm(c - (p + a*h), axis=-1) for a in ARM], axis=0) - ARM_R - r
    return np.minimum(rect, arm)


def detail(path, ep):
    sc, z = scenario(path), np.load(path/"trajectory.npz")
    mv = np.array(ep["obstacle_is_moving"], bool)
    t0, j = np.argwhere(z["contact_obstacle"][:, mv].astype(bool))[0]
    k = np.flatnonzero(mv)[j]
    r = sc["obstacles"][k]["r"]
    p, c = z["base_m"][:, :2], z["obstacle_xy"][:, k]
    yaw = np.unwrap(np.arctan2(z["observations"][:, 3], z["observations"][:, 4]))
    vel = np.gradient(p, DT, axis=0)
    v = (vel*np.stack([np.cos(yaw), np.sin(yaw)], 1)).sum(1)          # forward speed
    w = np.gradient(yaw, DT)
    a = max(t0 - 200, 0)
    out = {"id": ep["path"]}
    for s in (4, 3, 2, 1, 0.5):
        i = max(t0 - int(s/DT), 0)
        out[f"v_{s}s"] = round(float(v[max(i-10, 0):i+10].mean()), 2)
    out["turn_4s_deg"] = round(float(np.degrees(yaw[t0] - yaw[max(t0 - 400, 0)])))
    out["gap_2s_m"] = round(float(gap(p[a], yaw[a], c[a], r)), 2)
    # counterfactuals from a = 2 s before contact, over the following 3 s of the mover's real path
    ts = np.arange(0, min(300, len(c) - a))*DT
    fut = c[a:a + len(ts)]
    h0 = np.array([np.cos(yaw[a]), np.sin(yaw[a])])
    for name, (vv, ww) in {"keep": (v[a], w[a]), "stop": (0.0, 0.0), "full": (0.25, 0.0), "reverse": (-0.1, 0.0)}.items():
        th = yaw[a] + ww*ts
        if abs(ww) > 1e-6:
            pts = p[a] + np.stack([(np.sin(th) - np.sin(yaw[a]))/ww, -(np.cos(th) - np.cos(yaw[a]))/ww], 1)*vv
        else:
            pts = p[a] + h0*vv*ts[:, None]
        out[f"cf_{name}"] = round(float(gap(pts, th, fut, r).min()), 2)
    out["actual_min_gap_3s"] = round(float(gap(p[a:a+len(ts)], yaw[a:a+len(ts)], fut, r).min()), 2)
    out["slowed"] = out["v_0.5s"] < 0.6*max(out["v_3s"], 0.05)
    return out


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    kinds = {h["id"]: h for h in json.loads((folder.parent/f"{folder.name}-fail-types.json").read_text())}
    eps = {e["path"]: e for e in json.loads((folder/"manifest.json").read_text())["episodes"]}
    rows = []
    for i, h in kinds.items():
        if h["type"] in ("mover hit: mover ran into moving robot", "mover hit: robot stopped, mover walked into it"):
            d = detail(folder/i, eps[i])
            d["side"], d["bearing_deg"], d["mover_dir_deg"], d["mover_speed"] = h["side"], h["bearing_deg"], h["mover_dir_deg"], h["mover_speed"]
            rows.append(d)
            print(d)
    n = len(rows)
    print(n, "crossing hits")
    for name in ("keep", "stop", "full", "reverse"):
        print(f"from 2 s before: {name:8s} would have missed {sum(r[f'cf_{name}'] > 0 for r in rows)}/{n}")
    print("missed by at least one simple option", sum(any(r[f"cf_{x}"] > 0 for x in ("keep", "stop", "full", "reverse")) for r in rows))
    print("slowed down in the last 3 s", sum(r["slowed"] for r in rows), "| speed 3 s before median", np.median([r["v_3s"] for r in rows]),
          "| 0.5 s before", np.median([r["v_0.5s"] for r in rows]))
    print("hit while slowed and 'keep' would have missed", sum(r["slowed"] and r["cf_keep"] > 0 for r in rows))
    (folder.parent/f"{folder.name}-crossing-detail.json").write_text(json.dumps(rows, indent=1, default=float))
