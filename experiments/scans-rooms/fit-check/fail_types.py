"""Every validation run of a room model that did not park, sorted into failure types (4 Nov). Read-only on saved
runs; uses rescored.json (parked rule) and hit_types.first_hit for mover hits. Usage: python fail_types.py FOLDER"""
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path("/home/akhjd/ros2_ws/comp0245_project/runs/scans-rooms/fit-check")))
from hit_types import first_hit, kind

DT = 0.01
FACING = np.radians(30)


def scenario(path):
    s = json.loads((path/"episode.json").read_text())["scenario"]
    return s if isinstance(s, dict) else eval(s, {"__builtins__": {}}, {"inf": float("inf")})


def moved(p, t):
    """Base speed over the 0.2 s ending at row t."""
    a = max(t - 20, 0)
    return float(np.linalg.norm(p[t] - p[a])/((t - a)*DT + 1e-9))


def mover_detail(path, ep):
    """Blind-zone facts for a mover hit: seconds since a ray last hit it, its surface distance from the robot centre then
    (the lidar starts 0.5 m from the centre)."""
    sc, z = scenario(path), np.load(path/"trajectory.npz")
    mv = np.array(ep["obstacle_is_moving"], bool)
    t0, j = np.argwhere(z["contact_obstacle"][:, mv].astype(bool))[0]
    k = np.flatnonzero(mv)[j]
    d = np.linalg.norm(z["obstacle_xy"][:, k] - z["base_m"][:, :2], axis=1) - sc["obstacles"][k]["r"]
    seen = np.flatnonzero(z["obstacle_seen"][:t0 + 1, k])
    if not len(seen):
        return {"last_seen_s_before": None, "last_seen_surface_m": None}
    return {"last_seen_s_before": round((t0 - seen[-1])*DT, 2), "last_seen_surface_m": round(float(d[seen[-1]]), 2)}


def other_contact(path, ep):
    """First wall / still-obstacle contact: when, robot speed, arm or base (arm discs reach 0.73 m ahead), target distance."""
    sc, z = scenario(path), np.load(path/"trajectory.npz")
    mv = np.array(ep["obstacle_is_moving"], bool)
    still = z["contact_obstacle"][:, ~mv].astype(bool).any(axis=1) if (~mv).any() else np.zeros(len(z["time_s"]), bool)
    wall = z["contact_wall"].astype(bool)
    t0 = int(np.flatnonzero(still | wall)[0])
    p = z["base_m"][:, :2]
    lid = z["lidar"][t0]
    yaw = np.arctan2(z["observations"][t0, 3], z["observations"][t0, 4])
    ang = np.degrees(np.linspace(-np.pi, np.pi, len(lid), endpoint=False))   # approximate: nearest ray direction
    return {"what": "wall" if wall[t0] else "still obstacle", "t_s": round(t0*DT, 1), "robot_speed": round(moved(p, t0), 2),
            "to_target_m": round(float(np.linalg.norm(p[t0] - np.array(sc["target"][:2]))), 2),
            "nearest_ray_m": round(float(lid.min()), 2), "contact_s": round(float((still | wall).sum()*DT), 2)}


def not_parked(path, s):
    """Contact-free but not parked: which part of the rule failed, and what the robot did at the end."""
    z = np.load(path/"trajectory.npz")
    p = z["base_m"][:, :2]
    yaw = np.unwrap(np.arctan2(z["observations"][:, 3], z["observations"][:, 4]))
    speed = np.linalg.norm(np.diff(p, axis=0), axis=1)/DT
    last5 = speed[-500:]
    moving_rows = np.flatnonzero(speed > 0.05)
    out = {"distance_error_m": round(s["distance_error_m"], 2), "range_m": round(s.get("range_m", np.nan), 2),
           "in_reach_area": s["in_reach_area"], "facing_deg": round(float(np.degrees(s["facing_error_rad"]))),
           "max_speed_last2s": round(s["max_speed_m_s"], 3), "max_yaw_rate_last2s": round(s["max_yaw_rate_m_s"] if "max_yaw_rate_m_s" in s else s["max_yaw_rate_rad_s"], 3),
           "mean_speed_last5s": round(float(last5.mean()), 3), "turned_last5s_deg": round(float(np.degrees(abs(yaw[-1] - yaw[-500])))),
           "last_moved_s": round(float(moving_rows[-1]*DT), 1) if len(moving_rows) else 0.0,
           "path_m": round(s["base_path_length_m"], 1), "nearest_ray_end_m": round(float(z["lidar"][-1].min()), 2)}
    if s["distance_error_m"] > 1.0:
        out["type"] = ("stuck: stopped early, far from target" if out["last_moved_s"] < 60 else
                       "lost: still moving at the end, far from target")
    elif not s["in_reach_area"] or s["facing_error_rad"] > FACING:
        out["type"] = "close but outside the rule: " + ", ".join(
            x for x, bad in (("reach area", not s["in_reach_area"]), ("facing", s["facing_error_rad"] > FACING)) if bad)
    else:
        out["type"] = "right place, not still in the last 2 s"
    return out


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    scores = {e["id"]: e for e in json.loads((folder/"rescored.json").read_text())["episodes"]}
    rows = []
    for ep in json.loads((folder/"manifest.json").read_text())["episodes"]:
        path, s = folder/ep["path"], scores[ep["scenario_id"]]
        if s["parked"]:
            continue
        hit = first_hit(path, ep["obstacle_is_moving"])
        if hit:
            who, side = kind(hit)
            row = {"id": ep["path"], "type": f"mover hit: {who}", "side": side, **hit, **mover_detail(path, ep)}
        elif s["contact_steps"] > 0:
            c = other_contact(path, ep)
            row = {"id": ep["path"], "type": f"{c['what']} hit", **c}
        else:
            row = {"id": ep["path"], **not_parked(path, s)}
        rows.append(row)
        print(row, flush=True)
    print()
    print(len(rows), "runs not parked")
    for t, n in Counter(r["type"] for r in rows).most_common():
        print(f"{n:3d}  {t}")
    (folder.parent/f"{folder.name}-fail-types.json").write_text(json.dumps(rows, indent=1, default=float))
