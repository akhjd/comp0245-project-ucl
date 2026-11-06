"""Crossing-mover training scenarios (4 Nov): the 200 room TRAINING layouts, each with one extra mover timed to cut
across the robot's way from the side or from behind-and-to-the-side, the situation behind most of window r3's mover
hits. Training split only (validation and test layouts are never used). Labelled by the window expert in DAgger.

Per layout: a crossing point C at 30-75% of the way from the start to the parking spot; the robot is assumed to get
there at 0.12-0.2 m/s after a 3 s start; the new mover (0.15-0.3 m/s, radius 0.15-0.27 m) heads for C from 60-160 deg
off the robot's direction of travel (either side) and reaches it at about the same time, then carries on to the far
wall and turns round there."""
import json
from pathlib import Path
import numpy as np

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")


def free_run(p, d, room, r):
    """How far a disc of radius r at p can go along unit direction d before touching the room's walls."""
    x0, y0, x1, y1 = room
    t = np.inf
    for k, (lo, hi) in enumerate(((x0, x1), (y0, y1))):
        if d[k] > 1e-9:
            t = min(t, (hi - r - p[k])/d[k])
        elif d[k] < -1e-9:
            t = min(t, (lo + r - p[k])/d[k])
    return t


def crossing(s, rng, robot_speed=(0.12, 0.2)):
    """One crossing mover for scenario s, timed for a robot driving at robot_speed (m/s, uniform); None if it does not
    fit. The 5 Nov overnight run uses (0.12, 0.35): the faster teacher cruises at a median 0.16 m/s (0.08-0.36)."""
    a, b = np.array(s["start"][:2]), np.array(s["park"])
    room = s["room"]
    path = b - a
    length = np.linalg.norm(path)
    if length < 1.5:
        return None
    heading = np.arctan2(path[1], path[0])
    for _ in range(50):
        f = rng.uniform(0.3, 0.75)
        c = a + f*path
        t_c = 3.0 + f*length/rng.uniform(*robot_speed)
        u, r = rng.uniform(0.15, 0.3), rng.uniform(0.15, 0.27)
        off = np.radians(rng.uniform(60, 120) if rng.random() < 0.7 else rng.uniform(120, 160))*rng.choice([-1, 1])
        out = np.array([np.cos(heading + off), np.sin(heading + off)])       # from C towards the mover's start
        dist = float(np.clip(u*t_c, 1.2, free_run(c, out, room, r + 0.05)))
        if dist < 1.2:
            continue
        m0 = c + dist*out
        if np.linalg.norm(m0 - a) < 1.3 or np.linalg.norm(m0 - b) < 1.0:
            continue
        if any(np.linalg.norm(m0 - np.array(o["xy"])) < o["r"] + r + 0.3 for o in s["obstacles"]):
            continue
        d = -out                                                              # direction of travel, through C
        ahead, behind = free_run(m0, d, room, r + 0.05), free_run(m0, -d, room, r + 0.05)
        if ahead < dist + 1.0:
            continue
        sweep = [-float(min(max(behind, 0.0), 0.5)), float(ahead)]
        return {"xy": [round(float(m0[0]), 4), round(float(m0[1]), 4)], "r": round(float(r), 3),
                "vel": [round(float(u*d[0]), 4), round(float(u*d[1]), 4)], "span": round(float(ahead), 3), "sweep": sweep,
                "crossing": True}
    return None


if __name__ == "__main__":
    # 5 Nov: optional arguments SOURCE OUT SEED for the new training rooms (default: the 4 Nov files, unchanged)
    import sys
    source, path, seed = (sys.argv[1:4] if len(sys.argv) > 3 else
                          ("scenarios_rooms_train.json", Path(__file__).parent/"scenarios_crossing_train.json", 20261104))
    rng = np.random.default_rng(int(seed))
    out = []
    for s in json.loads((ROOT/source).read_text()):
        assert s["split"] == "train"
        m = crossing(s, rng)
        if m is None:
            continue
        out.append({**s, "id": s["id"].replace("train", "traincross"), "obstacles": s["obstacles"] + [m],
                    "seed": int(rng.integers(1_000_000)), "expert": "window"})
    path = ROOT/path
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(out, indent=1))
    print(len(out), "crossing scenarios ->", path)
