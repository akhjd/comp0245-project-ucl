"""Check preprocess.py's features against the truth on saved expert runs (plan step 2). Reads only; trains nothing.

    python tests/check_preprocess.py --runs-per-case 4 --out runs/preprocess_check.json

The truth comes from each scenario file and the saved obstacle positions: walls and furniture boxes, doorways (the gap between
two wall pieces in line), moving and still cylinders. It is used only here, to measure the features, never as an MLP input.
Measured on every 5th row after the first second, with things counted only when the lidar could see them (within 3 m and with
nothing in the way).
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import preprocess as pp
from train_mlp import CASES, state_rows

SEE_M, MATCH_M = 3.0, 0.5


def first_hit(o, p, boxes, circles):
    """Distance along the segment o -> p to the first box or circle, or inf."""
    d = p - o
    length = np.linalg.norm(d)
    if length < 1e-9:
        return np.inf
    d = d/length
    best = np.inf
    for x0, y0, x1, y1 in boxes:
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (np.array([x0, y0]) - o)/d, (np.array([x1, y1]) - o)/d
        lo, hi = np.nanmax(np.minimum(t1, t2)), np.nanmin(np.maximum(t1, t2))
        if hi >= max(lo, 0) and lo <= length:
            best = min(best, max(lo, 0))
    for cx, cy, r in circles:
        f = o - (cx, cy)
        b, c = f @ d, f @ f - r*r
        if b*b - c >= 0:
            t = -b - np.sqrt(b*b - c)
            if 0 <= t <= length:
                best = min(best, t)
    return best


def true_doors(boxes):
    """Doorways: the gap between two wall pieces on the same line. Returns [(middle, direction of the wall)]."""
    walls = [b for b in boxes if b.get("kind") != "furniture"]
    doors = []
    for axis in (0, 1):                     # axis 0: a wall running along y (thin in x)
        lines = {}
        for b in walls:
            lo, hi = np.array(b["min"]), np.array(b["max"])
            if hi[axis] - lo[axis] < 0.2:
                lines.setdefault(round((lo[axis] + hi[axis])/2, 3), []).append((lo[1 - axis], hi[1 - axis]))
        for at, pieces in lines.items():
            pieces.sort()
            for (_, end), (start, _) in zip(pieces, pieces[1:]):
                if 1.0 <= start - end <= 2.0:
                    middle = [0, 0]
                    middle[axis], middle[1 - axis] = at, (start + end)/2
                    along = np.zeros(2)
                    along[1 - axis] = 1.0
                    doors.append((np.array(middle), along))
    return doors


def to_world(pose, d, si, co):
    x, y, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([x + d*(c*co - s*si), y + d*(s*co + c*si)])


def check_run(folder, tally):
    z = np.load(folder/"trajectory.npz")
    summary = json.loads((folder/"summary.json").read_text())
    scenario = json.loads((folder/"episode.json").read_text())["scenario"]
    obs, t, lidar = z["observations"], z["time_s"], z["lidar"]
    targets = np.array(scenario.get("targets", [scenario["target"]]))
    slots, _ = state_rows(obs, t, targets, {p["dot"]: p["time_s"] for p in summary["picks"]})
    dots = slots[:, 7:].reshape(-1, 3, 4)
    yaw = np.arctan2(obs[:, 3], obs[:, 4])
    room = scenario.get("room", [-4, -4, 4, 4])
    boxes = [(*b["min"], *b["max"]) for b in scenario.get("boxes", [])]
    x0, y0, x1, y1 = room
    walls_room = [(x0 - 0.1, y0, x0, y1), (x1, y0, x1 + 0.1, y1), (x0, y0 - 0.1, x1, y0), (x0, y1, x1, y1 + 0.1)]
    doors = true_doors(scenario.get("boxes", []))
    moving = np.array([np.any(np.asarray(o.get("vel", [0, 0])) != 0) for o in scenario["obstacles"]], bool)
    radius = np.array([o["r"] for o in scenario["obstacles"]])
    if "obstacle_xy" in z.files:
        xy = z["obstacle_xy"]
    else:                                  # saved only when something moves: still cylinders stay at their start
        if moving.any():
            raise ValueError(f"{folder}: movers but no saved positions")
        xy = np.repeat(np.array([o["xy"] for o in scenario["obstacles"]], float).reshape(1, -1, 2), len(t), axis=0)
    velocity = np.gradient(xy, t, axis=0) if len(t) > 1 and xy.shape[1] else np.zeros_like(xy)
    case = scenario["case"]
    pre = pp.Preprocessor()
    begin = time.time()
    for i in range(len(t)):
        pose = (obs[i, 0], obs[i, 1], yaw[i])
        f = pre.step(t[i], lidar[i], pose, dots[i])
        if t[i] < 1.0 or i % 5:
            continue
        v = dict(zip(pp.NAMES, f))
        robot = obs[i, :2]
        circles = [(*xy[i, k], radius[k]) for k in range(xy.shape[1])]
        solid = boxes + walls_room
        # Openings and doorways, recomputed with the same functions the features use
        groups, pts = pp.clusters(np.clip(lidar[i], 0, pp.RANGE))
        gaps = pp.openings(groups, np.clip(lidar[i], 0, pp.RANGE), pts)
        found = [to_world(pose, *pp.polar(m)) for m, *_ in gaps]
        doorish = [g for g, (_, _, door, *_) in zip(found, gaps) if door]
        tally[case]["openings_found"] += len(found)
        tally[case]["openings_at_door"] += sum(any(np.linalg.norm(g - d) < MATCH_M for d, _ in doors) for g in found)
        tally[case]["door_like_found"] += len(doorish)
        tally[case]["door_like_at_door"] += sum(any(np.linalg.norm(g - d) < MATCH_M for d, _ in doors) for g in doorish)
        for d, along in doors:
            view = (d - robot)/max(np.linalg.norm(d - robot), 1e-9)
            facing = abs(view @ along) < np.cos(np.radians(30))           # not looking along the wall: rays can pass through
            if np.linalg.norm(d - robot) < SEE_M and facing and first_hit(robot, d, solid, circles) == np.inf:
                tally[case]["doors_visible"] += 1
                tally[case]["doors_found"] += any(np.linalg.norm(g - d) < MATCH_M for g in found)
                tally[case]["doors_found_door_like"] += any(np.linalg.norm(g - d) < MATCH_M for g in doorish)
        tally[case]["rows"] += 1
        # Walls: nearest wall or furniture or room wall
        gap = min([np.hypot(max(bx0 - robot[0], 0, robot[0] - bx1), max(by0 - robot[1], 0, robot[1] - by1))
                   for bx0, by0, bx1, by1 in solid])
        if gap < SEE_M:
            tally[case]["wall_near_rows"] += 1
            tally[case]["wall_seen"] += v["wall_seen"]
            if v["wall_seen"]:
                tally[case]["wall_errors"].append(abs(v["wall_distance"] - gap))
        # Line to each dot
        for d in range(3):
            if dots[i, d, 3] < 0.5:
                continue
            target = targets[d, :2]
            distance = np.linalg.norm(target - robot)
            truly = first_hit(robot, target, solid, circles) < min(distance, pp.HIT) - pp.BLOCK_MARGIN_M
            said = bool(v[f"dot{d}_blocked"])
            tally[case]["dot_rows"] += 1
            tally[case]["dot_truly_blocked"] += truly
            tally[case]["dot_agree"] += truly == said
            tally[case]["dot_blocked_found"] += truly and said
            tally[case]["dot_false_blocked"] += (not truly) and said
        # Movers
        near_movers = [k for k in np.flatnonzero(moving) if np.linalg.norm(xy[i, k] - robot) < SEE_M
                       and first_hit(robot, xy[i, k], solid, []) >= np.linalg.norm(xy[i, k] - robot) - radius[k]]
        said_at = to_world(pose, v["mover_distance"], v["mover_sin"], v["mover_cos"]) if v["mover_seen"] else None
        if near_movers:
            tally[case]["mover_rows"] += 1
            k = min(near_movers, key=lambda k: np.linalg.norm(xy[i, k] - robot))
            if said_at is not None and np.linalg.norm(said_at - xy[i, k]) < MATCH_M + radius[k]:
                tally[case]["mover_found"] += 1
                c, s = np.cos(yaw[i]), np.sin(yaw[i])
                u = np.array([c*v["mover_v_forward"] - s*v["mover_v_left"], s*v["mover_v_forward"] + c*v["mover_v_left"]])
                tally[case]["mover_speed_errors"].append(float(np.linalg.norm(u - velocity[i, k])))
        if said_at is not None and not any(np.linalg.norm(said_at - xy[i, k]) < MATCH_M + radius[k] for k in np.flatnonzero(moving)):
            tally[case]["mover_false"] += 1
        tally[case]["mover_said"] += said_at is not None
    tally[case]["seconds"] += time.time() - begin
    tally[case]["steps"] += len(t)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(ROOT/"runs/dataset"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--runs-per-case", type=int, default=4)
    parser.add_argument("--out", default=str(ROOT/"runs/preprocess_check.json"))
    args = parser.parse_args()
    keys = ["rows", "openings_found", "openings_at_door", "doors_visible", "doors_found", "wall_near_rows", "wall_seen", "dot_rows",
            "dot_truly_blocked", "dot_agree", "dot_blocked_found", "dot_false_blocked", "mover_rows", "mover_found", "mover_false",
            "mover_said", "seconds", "steps", "door_like_found", "door_like_at_door", "doors_found_door_like"]
    tally = {c: {**{k: 0 for k in keys}, "wall_errors": [], "mover_speed_errors": []} for c in CASES}
    for case in CASES:
        folders = [Path(n).parent for n in sorted(glob.glob(f"{args.data}/{case}/part*/*/trajectory.npz"))]
        folders = [f for f in folders if json.loads((f/"episode.json").read_text())["scenario"]["split"] == args.split]
        has_mover = lambda f: any(np.any(np.asarray(o.get("vel", [0, 0])) != 0)
                                  for o in json.loads((f/"episode.json").read_text())["scenario"]["obstacles"])
        with_m, without = [f for f in folders if has_mover(f)], [f for f in folders if not has_mover(f)]
        pick = lambda fs, n: fs[::max(len(fs)//max(n, 1), 1)][:n]
        chosen = pick(with_m, (args.runs_per_case + 1)//2) if with_m else []
        chosen += pick(without, args.runs_per_case - len(chosen))
        for f in chosen:            # half with a mover where the case has movers, so the mover numbers mean something
            check_run(f, tally)
        print(f"{case}: {args.runs_per_case} runs done", flush=True)
    share = lambda a, b: round(a/b, 3) if b else None
    report = {}
    for c, s in tally.items():
        report[c] = {
            "rows_checked": s["rows"],
            "doorways_found": share(s["doors_found"], s["doors_visible"]), "doorways_visible": s["doors_visible"],
            "openings_per_row": share(s["openings_found"], s["rows"]), "openings_at_a_doorway": share(s["openings_at_door"], s["openings_found"]),
            "wall_seen_when_within_3m": share(s["wall_seen"], s["wall_near_rows"]),
            "wall_distance_error_median_m": round(float(np.median(s["wall_errors"])), 3) if s["wall_errors"] else None,
            "dot_line_agrees": share(s["dot_agree"], s["dot_rows"]), "dot_truly_blocked_share": share(s["dot_truly_blocked"], s["dot_rows"]),
            "dot_blocked_caught": share(s["dot_blocked_found"], s["dot_truly_blocked"]),
            "dot_false_blocked": share(s["dot_false_blocked"], s["dot_rows"] - s["dot_truly_blocked"]),
            "mover_found_when_visible": share(s["mover_found"], s["mover_rows"]), "mover_rows": s["mover_rows"],
            "mover_speed_error_median_m_s": round(float(np.median(s["mover_speed_errors"])), 3) if s["mover_speed_errors"] else None,
            "mover_false_share_of_reports": share(s["mover_false"], s["mover_said"]),
            "mover_false_per_row": share(s["mover_false"], s["rows"]),
            "door_like_at_a_doorway": share(s["door_like_at_door"], s["door_like_found"]),
            "doorways_found_door_like": share(s["doors_found_door_like"], s["doors_visible"]),
            "microseconds_per_scan": round(1e6*s["seconds"]/max(s["steps"], 1)),
        }
    Path(args.out).write_text(json.dumps(report, indent=1))
    for c, r in report.items():
        print(c, json.dumps(r))


if __name__ == "__main__":
    main()
