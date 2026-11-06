"""What kinds of mover hits does a room model have? (4 Nov.) For every validation run with a mover contact: the
first contact, who was moving into whom, from which side, which part of the robot, how far from the target, and how
long the mover had been in view. Read-only on saved runs. Usage: python hit_types.py FOLDER [EXPERT_FOLDER]"""
import ast
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

ARM, ARM_R, BASE_R = np.array([0.46, 0.56, 0.66]), 0.07, 0.47     # obstacle_env: arm discs ahead of the base centre


def first_hit(folder, moving):
    ep = json.loads((folder/"episode.json").read_text())
    sc = ep["scenario"] if isinstance(ep["scenario"], dict) else ast.literal_eval(ep["scenario"])
    z = np.load(folder/"trajectory.npz")
    dt = 0.001*int(ep.get("saved_every", 10))
    moving = np.array(moving, bool)
    radius = np.array([o.get("r", 0.0) for o in sc["obstacles"]])
    assert len(radius) == len(moving) == z["obstacle_xy"].shape[1], folder
    hit = z["contact_obstacle"][:, moving].astype(bool) if moving.any() else np.zeros((len(z["time_s"]), 0), bool)
    if not hit.any():
        return None
    t0, j = np.argwhere(hit)[0]
    k = np.flatnonzero(moving)[j]
    p, xy = z["base_m"][:, :2], z["obstacle_xy"][:, k]
    yaw = np.arctan2(z["observations"][:, 3], z["observations"][:, 4])
    lag = max(1, int(0.2/dt))
    a = max(t0 - lag, 0)
    v_robot, v_mover = (p[t0] - p[a])/((t0 - a)*dt + 1e-9), (xy[t0] - xy[a])/((t0 - a)*dt + 1e-9)
    rel = xy[t0] - p[t0]
    u = rel/np.linalg.norm(rel)
    robot_in, mover_in = float(v_robot @ u), float(-(v_mover @ u))        # closing speeds along the line of centres
    heading = np.array([np.cos(yaw[t0]), np.sin(yaw[t0])])
    bearing = np.degrees(np.arctan2(heading[0]*rel[1] - heading[1]*rel[0], heading @ rel))
    mover_dir = np.degrees(np.arctan2(heading[0]*v_mover[1] - heading[1]*v_mover[0], heading @ v_mover))
    arm = p[t0] + ARM[:, None]*heading
    arm_gap = (np.linalg.norm(arm - xy[t0], axis=1) - ARM_R - radius[k]).min()
    base_gap = np.linalg.norm(rel) - BASE_R - radius[k]
    seen = z["obstacle_seen"][:t0 + 1, k].astype(bool)
    in_view = (t0 - np.flatnonzero(~seen)[-1])*dt if (~seen).any() else t0*dt
    near = np.linalg.norm(xy[:t0 + 1] - p[:t0 + 1], axis=1) < 1.5
    close_for = (t0 - np.flatnonzero(~near)[-1])*dt if (~near).any() else t0*dt
    speed_before = np.linalg.norm(np.diff(p[max(t0 - int(2/dt), 0):t0 + 1], axis=0), axis=1)/dt
    turn_before = np.degrees(np.angle(np.exp(1j*(yaw[t0] - yaw[max(t0 - int(2/dt), 0)]))))
    target = np.array(sc["target"][:2])
    return {"id": sc["id"], "t_s": round(t0*dt, 1), "robot_speed": round(float(np.linalg.norm(v_robot)), 2),
            "robot_closing": round(robot_in, 2), "mover_closing": round(mover_in, 2),
            "bearing_deg": round(float(bearing)), "mover_dir_deg": round(float(mover_dir)),
            "part": "arm" if arm_gap < base_gap else "base",
            "to_target_m": round(float(np.linalg.norm(p[t0] - target)), 2),
            "in_view_s": round(float(in_view), 1), "within_1p5m_s": round(float(close_for), 1),
            "speed_2s_before": round(float(speed_before.mean()), 2) if len(speed_before) else 0.0,
            "min_speed_2s_before": round(float(speed_before.min()), 2) if len(speed_before) else 0.0,
            "turn_2s_deg": round(float(turn_before)), "contact_s": round(float(hit[:, j].sum()*dt), 2),
            "mover_speed": round(float(np.linalg.norm(v_mover)), 2)}


def kind(h):
    """Plain-language hit type."""
    side = "front" if abs(h["bearing_deg"]) < 45 else ("behind" if abs(h["bearing_deg"]) > 135 else "side")
    if h["robot_speed"] < 0.03:
        who = "robot stopped, mover walked into it"
    elif h["robot_closing"] > h["mover_closing"]:
        who = "robot drove into mover"
    else:
        who = "mover ran into moving robot"
    return who, side


if __name__ == "__main__":
    folder = Path(sys.argv[1])

    def runs(folder):
        return [(folder/e["path"], e.get("obstacle_is_moving", []))
                for e in json.loads((folder/"manifest.json").read_text())["episodes"]]
    hits = [h for h in (first_hit(f, m) for f, m in runs(folder)) if h]
    expert = {f.name: first_hit(f, m) is not None for f, m in runs(Path(sys.argv[2]))} if len(sys.argv) > 2 else {}
    for h in hits:
        who, side = kind(h)
        h["who"], h["side"] = who, side
        h["expert_hit_here"] = expert.get(h["id"])
        print(h)
    print(len(hits), "runs with a mover hit")
    print("who", Counter(h["who"] for h in hits))
    print("side", Counter(h["side"] for h in hits))
    print("who x side", Counter((h["who"], h["side"]) for h in hits))
    print("part", Counter(h["part"] for h in hits))
    print("near target (<1 m)", sum(h["to_target_m"] < 1.0 for h in hits), "| mid-route", sum(h["to_target_m"] >= 1.0 for h in hits))
    print("mover in view < 1 s before hit", sum(h["in_view_s"] < 1.0 for h in hits))
    print("robot slowed to < 0.03 m/s in the 2 s before", sum(h["min_speed_2s_before"] < 0.03 for h in hits))
    (folder.parent/f"{folder.name}-hit-types.json").write_text(json.dumps(hits, indent=1))
