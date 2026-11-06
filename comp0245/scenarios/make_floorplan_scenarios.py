"""Write floor-plan scenarios for comp0245/training/collect_obstacles.py: hand-made rooms with
interior walls, doorways and furniture.

--layouts random draws a new random floor plan for each scenario. Interior walls and furniture are
axis-aligned "boxes". Doorways are 1.4 m wide (the robot needs about 1.2 m at its normal
clearance). Each scenario has a random start and target; --cross-rooms (default share 0.7) of them
must have no straight line of sight, so the robot has to use a doorway. --mover-share of them add
one moving cylinder from the layout's list. Every scenario has a route (FloorPlanRoute), and movers
keep off the start, parking spot and target.

Example:
    python -m comp0245.scenarios.make_floorplan_scenarios --out runs/floorplan.json --count 30 \\
        --seed 1
    python -m comp0245.training.collect_obstacles --scenarios runs/floorplan.json --out runs/fp1 \\
        --only fp_000 --rrt --duration 110 --gui
"""

import argparse
import json
from pathlib import Path

import numpy as np

from comp0245.scenarios.make_obstacle_scenarios import segment_distance
from comp0245.sim.obstacle_env import ROBOT_RADIUS, FloorPlanRoute, box_gaps, standoff


def wall(x0, y0, x1, y1):
    return {"min": [x0, y0], "max": [x1, y1], "kind": "wall"}


def couch(x0, y0, x1, y1, h=0.8):
    return {"min": [x0, y0], "max": [x1, y1], "kind": "furniture", "h": h}


ROOM = [-4.0, -4.0, 4.0, 4.0]
LAYOUTS = {
    # one wall with a doorway; a couch on the left, a couch and a table on the right
    "two_rooms": {
        "boxes": [
            wall(-0.05, -4, 0.05, -0.7),
            wall(-0.05, 0.7, 0.05, 4),
            couch(-3.2, 1.8, -1.4, 2.8),
            couch(1.5, -3.2, 3.4, -2.2),
            couch(1.6, 1.6, 2.6, 2.6, h=0.7),
        ],
        "movers": [
            {"xy": [1.0, -3.0], "r": 0.2, "vel": [0.0, 0.2], "sweep": [-0.4, 6.4]},
            {"xy": [-1.0, 3.0], "r": 0.2, "vel": [0.0, -0.2], "sweep": [-0.4, 6.4]},
        ],
    },
    "three_rooms": {  # a living room on top, two rooms below, joined by doorways
        "boxes": [
            wall(-4, 0.95, -0.7, 1.05),
            wall(0.7, 0.95, 4, 1.05),
            wall(1.45, -4, 1.55, -2.4),
            wall(1.45, -1.0, 1.55, 0.95),
            couch(-3.0, 2.4, -1.0, 3.4),
            couch(1.0, 2.6, 3.0, 3.4),
            couch(-3.2, -2.0, -2.2, 0.0),
            couch(2.5, -3.4, 3.6, -1.5),
        ],
        "movers": [
            {"xy": [-3.0, 1.8], "r": 0.2, "vel": [0.2, 0.0], "sweep": [-0.5, 6.5]},
            {"xy": [-0.2, -3.0], "r": 0.2, "vel": [0.0, 0.15], "sweep": [-0.5, 3.5]},
        ],
    },
}


def random_layout(rng):
    """Split the room with one wall, usually split one side again (each wall has a 1.4 m
    doorway), then place 2-4 pieces of furniture away from the doorways by rejection sampling.
    Movers run along clear straight lines."""
    x = rng.uniform(-1.5, 1.5)
    door = rng.uniform(-3.0, 3.0)
    boxes, doors = (
        [wall(x - 0.05, -4, x + 0.05, door - 0.7), wall(x - 0.05, door + 0.7, x + 0.05, 4)],
        [np.array([x, door])],
    )
    if rng.random() < 0.7:
        left = rng.random() < 0.5
        x0, x1 = (-4.0, x - 0.05) if left else (x + 0.05, 4.0)
        y = rng.uniform(-1.5, 1.5)
        gate = rng.uniform(x0 + 0.9, x1 - 0.9)
        if abs(y - door) > 1.5:  # keep the second wall clear of the first doorway
            boxes += [
                wall(x0, y - 0.05, gate - 0.7, y + 0.05),
                wall(gate + 0.7, y - 0.05, x1, y + 0.05),
            ]
            doors.append(np.array([gate, y]))
    for _ in range(int(rng.integers(2, 5))):
        for _ in range(50):
            size = rng.uniform([0.8, 0.8], [2.0, 1.1])[:: 1 if rng.random() < 0.5 else -1]
            lo = rng.uniform(-3.8, 3.8 - size)
            hi = lo + size
            others_lo = np.array([b["min"] for b in boxes], dtype=float)
            others_hi = np.array([b["max"] for b in boxes], dtype=float)
            corners = np.array([lo, hi, [lo[0], hi[1]], [hi[0], lo[1]], (lo + hi) / 2])
            if (
                box_gaps(corners, others_lo, others_hi).min() < 1.15
                or box_gaps(np.array(doors), lo[None], hi[None]).min() < 1.3
            ):
                continue
            boxes.append(couch(*map(float, lo), *map(float, hi), h=float(rng.uniform(0.6, 0.9))))
            break
    lo = np.array([b["min"] for b in boxes], dtype=float)
    hi = np.array([b["max"] for b in boxes], dtype=float)
    movers = []
    for _ in range(30):
        axis = int(rng.integers(2))
        d = np.zeros(2)
        d[axis] = rng.choice([-1, 1])
        c = rng.uniform(-3.4, 3.4, 2)
        r, speed = float(rng.uniform(0.15, 0.3)), float(rng.uniform(0.1, 0.3))
        along = np.arange(0.0, 7.0, 0.1)

        def reach(sign):
            pts = c + sign * along[:, None] * d
            ok = (box_gaps(pts, lo, hi).min(axis=1) >= r + 0.05) & np.all(
                np.abs(pts) <= 4 - r - 0.05, axis=1
            )
            return along[np.argmin(ok)] - 0.1 if not ok.all() else along[-1]

        if box_gaps(c, lo, hi).min() < r + 0.05:
            continue
        up, down = reach(1), reach(-1)
        if up + down >= 2.5 and up > 0.2:
            movers.append(
                {
                    "xy": [round(float(v), 3) for v in c],
                    "r": round(r, 3),
                    "vel": [round(float(v), 3) for v in d * speed],
                    "sweep": [round(float(-down), 3), round(float(up), 3)],
                }
            )
        if len(movers) == 4:
            break
    return {"boxes": boxes, "movers": movers}


def clear_of(point, lo, hi, gap):
    return bool(box_gaps(np.asarray(point, dtype=float), lo, hi).min() >= gap)


def sample(rng, name, args):
    layout = random_layout(rng) if name == "random" else LAYOUTS[name]
    lo = np.array([b["min"] for b in layout["boxes"]], dtype=float)
    hi = np.array([b["max"] for b in layout["boxes"]], dtype=float)
    route = FloorPlanRoute(ROOM, lo, hi, [], [])
    room_lo, room_hi = np.array(ROOM[:2]), np.array(ROOM[2:])
    want_cross = rng.random() < args.cross_rooms
    for _ in range(3000):
        start = rng.uniform(room_lo + 0.8, room_hi - 0.8)
        yaw = rng.uniform(-np.pi, np.pi)
        target = rng.uniform(room_lo + 0.3, room_hi - 0.3)
        z = rng.uniform(*args.height)
        bearing = rng.uniform(-np.pi, np.pi)
        ahead = np.array([np.cos(bearing), np.sin(bearing)])
        park = target - standoff(z) * ahead
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        if not (
            clear_of(start, lo, hi, ROBOT_RADIUS + 0.25)
            and clear_of(start + 0.66 * heading, lo, hi, 0.2)
        ):
            continue
        if (
            not clear_of(park, lo, hi, ROBOT_RADIUS + 0.2)
            or np.any(park < room_lo + ROBOT_RADIUS + 0.2)
            or np.any(park > room_hi - ROBOT_RADIUS - 0.2)
        ):
            continue
        if not all(
            clear_of(park + s * (target - park), lo, hi, 0.2) for s in np.linspace(0, 1, 8)
        ):
            continue
        if (
            np.linalg.norm(target - start) < 2.5
            or route.sees(start, park) == want_cross
            or route.path(start, park) is None
        ):
            continue
        targets, parks = [[*target, z]], [park]
        for _ in range(300):
            if len(targets) == args.dots:
                break
            t2 = rng.uniform(room_lo + 0.3, room_hi - 0.3)
            z2 = rng.uniform(*args.height)
            b2 = rng.uniform(-np.pi, np.pi)
            p2 = t2 - standoff(z2) * np.array([np.cos(b2), np.sin(b2)])
            if (
                min(np.linalg.norm(t2 - np.array(q[:2])) for q in targets) < 1.5
                or np.any(p2 < room_lo + ROBOT_RADIUS + 0.2)
                or np.any(p2 > room_hi - ROBOT_RADIUS - 0.2)
            ):
                continue
            if (
                clear_of(p2, lo, hi, ROBOT_RADIUS + 0.2)
                and all(clear_of(p2 + u * (t2 - p2), lo, hi, 0.2) for u in np.linspace(0, 1, 8))
                and route.path(start, p2) is not None
            ):
                targets.append([*t2, z2])
                parks.append(p2)
        if len(targets) < args.dots:
            continue
        obstacles = []
        if rng.random() < args.mover_share:
            if not layout["movers"]:
                continue
            m = layout["movers"][int(rng.integers(len(layout["movers"])))]
            d = np.array(m["vel"]) / np.linalg.norm(m["vel"])
            sweep = np.array(m["xy"]) + np.linspace(*m["sweep"], 40)[:, None] * d
            if np.linalg.norm(np.array(m["xy"]) - start) < 1.4 or any(
                np.linalg.norm(sweep - q, axis=1).min() < m["r"] + ROBOT_RADIUS + 0.35
                for q in parks
            ):
                continue
            if any(
                min(segment_distance(q, a, np.array(t[:2])) for q in sweep) < m["r"] + 0.15
                for a, t in zip(parks, targets, strict=False)
            ):
                continue
            obstacles.append(m)
        statics = []
        for _ in range(int(rng.integers(args.static_count[0], args.static_count[1] + 1))):
            for _ in range(100):
                c, r = rng.uniform(room_lo + 0.5, room_hi - 0.5), rng.uniform(*args.radius)
                if (
                    not clear_of(c, lo, hi, r + 1.1)
                    or np.linalg.norm(c - start) < r + ROBOT_RADIUS + 0.45
                ):
                    continue
                if any(np.linalg.norm(c - q) < r + ROBOT_RADIUS + 0.2 for q in parks):
                    continue
                if any(
                    segment_distance(c, a, np.array(t[:2])) < r + 0.15
                    for a, t in zip(parks, targets, strict=False)
                ):
                    continue
                if any(np.linalg.norm(c - np.array(o["xy"])) < r + o["r"] + 1.2 for o in statics):
                    continue
                if any(
                    min(
                        np.linalg.norm(c - q)
                        for q in np.array(m["xy"])
                        + np.linspace(*m["sweep"], 40)[:, None]
                        * np.array(m["vel"])
                        / np.linalg.norm(m["vel"])
                    )
                    < r + m["r"] + 0.3
                    for m in obstacles
                ):
                    continue
                statics.append({"xy": [round(float(v), 4) for v in c], "r": round(float(r), 3)})
                break
        if statics:  # the cylinders must leave a route to every spot
            with_statics = FloorPlanRoute(
                ROOM, lo, hi, [np.array(o["xy"]) for o in statics], [o["r"] for o in statics]
            )
            if any(with_statics.path(start, q) is None for q in parks):
                continue
            obstacles = obstacles + statics

        def rnd(v):
            return [round(float(x), 4) for x in v]

        return {
            "layout": name,
            "room": ROOM,
            "boxes": layout["boxes"],
            "fixed_park": True,
            "start": rnd([*start, yaw]),
            "target": rnd([*target, z]),
            "park": rnd(park),
            "obstacles": obstacles,
            **(
                {"targets": [rnd(t) for t in targets], "parks": [rnd(q) for q in parks]}
                if args.dots > 1
                else {}
            ),
        }
    return sample(
        rng, name, args
    )  # this random layout has no valid start and target: draw another


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layouts", nargs="+", default=list(LAYOUTS))
    parser.add_argument("--cross-rooms", type=float, default=0.7)
    parser.add_argument("--mover-share", type=float, default=0.5)
    parser.add_argument("--dots", type=int, default=1, help="red dots per scenario")
    parser.add_argument(
        "--static-count",
        type=int,
        nargs=2,
        default=[0, 0],
        metavar=("LO", "HI"),
        help="static cylinders per scenario",
    )
    parser.add_argument(
        "--radius", type=float, nargs=2, default=[0.15, 0.3], help="static cylinder radius, metres"
    )
    parser.add_argument(
        "--height", type=float, nargs=2, default=[0.35, 0.8], help="target height, metres"
    )
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Choose a new scenarios file: {out}")
    rng = np.random.default_rng(args.seed)
    scenarios = [
        {
            "id": f"fp_{i:03d}",
            "split": "train",
            **sample(rng, args.layouts[i % len(args.layouts)], args),
            "seed": int(rng.integers(1_000_000)),
        }
        for i in range(args.count)
    ]
    out.write_text(json.dumps(scenarios, indent=1))
    print(f"Wrote {len(scenarios)} scenarios to {out}")


if __name__ == "__main__":
    main()
