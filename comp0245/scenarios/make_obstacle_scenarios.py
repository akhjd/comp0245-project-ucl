"""Write a reproducible obstacle scenarios file for comp0245/training/collect_obstacles.py.

Each scenario has a random start pose, a target, the base parking spot ("park") and 1-3 cylinders.
Rules, so every scenario is solvable and nothing overlaps at the start:
- one "blocker" sits roughly on the straight line from the start to the parking spot, so avoidance
  is needed;
- obstacles keep clear of the start footprint, the parking spot, the target and the park-to-target
  line (the arm must not reach through an obstacle);
- any two obstacles leave a gap the robot fits through, so the field cannot trap it between them.
The parking spot is at the reach-zone distance (obstacle_env.standoff ->
reach_zone.parking_distance) along the target-to-start direction.
archive/scenarios_obstacles_v1.json was made with the course standoff rule instead.

--hard makes difficulty on purpose (see sample_hard_scenario): 2-3 blockers along the route,
sometimes a wall across it or an obstacle right beside the parking spot, and gaps down to HARD_GAP.
Not every hard layout is solvable by the expert: collect expert runs and keep only the successful
ones (the professor's advice).

--moving-count LO HI adds that many MOVING cylinders (sample_moving_scenario); --static-count LO HI
adds static ones beside them (0 0 = moving only, 1 3 = mixed). Each mover has "vel" [vx, vy] (m/s)
and "span" (m; it turns round after travelling that far from its start). It starts off the route
and is timed to cross the start-park line about when the robot gets there, so the robot has to wait
or avoid it. Movers never enter the parking spot or the arm's reach line. These layouts are not all
solvable: collect expert runs and keep the successful ones.

Example:
    python -m comp0245.scenarios.make_obstacle_scenarios --moving-count 2 3 --out \\
        moving_only.json --train 10 --validation 0 --test 0
    python -m comp0245.scenarios.make_obstacle_scenarios --moving-count 1 2 --static-count 1 3 \\
        --out mixed.json --train 10 --validation 0 --test 0
    python -m comp0245.scenarios.make_obstacle_scenarios --out scenarios_obstacles_v2.json \\
        --train 60 --validation 10 --test 10 --seed 0
    python -m comp0245.scenarios.make_obstacle_scenarios --hard --out hard.json --train 140 \\
        --validation 0 --test 0 --prefix hard2 \
        --distance 3.5 5 --radius 0.2 0.35 --max-obstacles 8 --seed 101
"""

import argparse
import json
from pathlib import Path

import numpy as np

from comp0245.sim.obstacle_env import ROBOT_RADIUS, standoff

ARM_START = np.array([0.0, 1.57, 0.0, 1.0, 0.0, 1.0, 0.0])  # the course's arm start pose (rad)


def segment_distance(c, a, b):
    """Distance from point c to the line segment a-b."""
    t = np.clip(np.dot(c - a, b - a) / np.dot(b - a, b - a), 0, 1)
    return float(np.linalg.norm(c - (a + t * (b - a))))


# m: narrowest gap between obstacles in --hard layouts (robot 0.63 m wide, 0.94 m turning circle)
HARD_GAP = 1.0


def fits(c, r, start, park, target, placed, gap=2 * ROBOT_RADIUS + 0.2, park_room=0.35):
    if (
        np.linalg.norm(c - start) < r + ROBOT_RADIUS + 0.45
    ):  # the first turn on the spot can slide ~0.4 m
        return False
    if np.linalg.norm(c - park) < r + ROBOT_RADIUS + park_room:  # room to manoeuvre onto the spot
        return False
    if segment_distance(c, park, target) < r + 0.15 or np.linalg.norm(c - target) < r + 0.3:
        return False
    return all(np.linalg.norm(c - c2) > r + r2 + gap for c2, r2 in placed)


def sample_scenario(rng, args):
    while True:
        start = rng.uniform(-1, 1, size=2)
        yaw = rng.uniform(-np.pi, np.pi)
        bearing = rng.uniform(-np.pi, np.pi)
        target_xy = start + rng.uniform(*args.distance) * np.array(
            [np.cos(bearing), np.sin(bearing)]
        )
        z = rng.uniform(*args.height)
        park = target_xy - standoff(z) * np.array([np.cos(bearing), np.sin(bearing)])
        placed = []
        # The blocker: somewhere in the middle of the start-park line, slightly off-centre.
        side = np.array([-np.sin(bearing), np.cos(bearing)])
        for _ in range(50):
            r = rng.uniform(*args.radius)
            # Never exactly on the line: a dead-centre obstacle leaves the expert no preferred
            # side.
            offset = rng.choice([-1, 1]) * rng.uniform(0.05, 0.2)
            c = start + rng.uniform(0.4, 0.6) * (park - start) + offset * side
            if fits(c, r, start, park, target_xy, placed):
                placed.append((c, r))
                break
        if not placed:
            continue
        # Extra obstacles anywhere around the route.
        lo, hi = np.minimum(start, target_xy) - 1.0, np.maximum(start, target_xy) + 1.0
        extra = rng.integers(0, args.max_obstacles)
        for _ in range(200):
            if len(placed) > extra:
                break
            r = rng.uniform(*args.radius)
            c = rng.uniform(lo, hi)
            if fits(c, r, start, park, target_xy, placed):
                placed.append((c, r))

        def rnd(v):
            return [round(float(x), 4) for x in v]

        return {
            "start": rnd([*start, yaw]),
            "target": rnd([*target_xy, z]),
            "park": rnd(park),
            "obstacles": [{"xy": rnd(c), "r": round(float(r), 3)} for c, r in placed],
        }


def sample_hard_scenario(rng, args):
    """A layout that is hard on purpose: half get a wall of 2-3 touching cylinders across the
    route; the rest get 2-3 blockers along it (walled layouts get any blockers that still fit);
    half an obstacle right beside the parking spot; then extra obstacles near the route up to
    --max-obstacles. Gaps between separate obstacles go down to HARD_GAP. Which features a
    layout gets is decided once, before any retries, so the halves stay halves."""
    want_wall, want_crowded = rng.random() < 0.5, rng.random() < 0.5
    while True:
        start = rng.uniform(-1, 1, size=2)
        yaw = rng.uniform(-np.pi, np.pi)
        bearing = rng.uniform(-np.pi, np.pi)
        ahead = np.array([np.cos(bearing), np.sin(bearing)])
        side = np.array([-np.sin(bearing), np.cos(bearing)])
        target_xy = start + rng.uniform(*args.distance) * ahead
        z = rng.uniform(*args.height)
        park = target_xy - standoff(z) * ahead
        route = park - start

        def ok(c, r, room=0.35):
            return fits(c, r, start, park, target_xy, placed, gap=HARD_GAP, park_room=room)

        placed = []
        # 1. Wall (half the layouts, placed first so the blockers fit around it): 2-3 cylinders
        # touching each other
        #    across the route, forcing a longer detour
        if want_wall:
            for _ in range(30):
                n, r = rng.integers(2, 4), rng.uniform(*args.radius)
                centre = start + rng.uniform(0.35, 0.65) * route + rng.uniform(-0.3, 0.3) * side
                wall = [(centre + (j - (n - 1) / 2) * 1.9 * r * side, r) for j in range(n)]
                if all(ok(c, r) for c, r in wall):
                    placed += wall
                    break
        walled = len(placed)
        if want_wall and not walled:
            continue
        # 2. Blockers: 2-3 along the route, each a little to one side, so the robot has to weave
        for f in np.sort(rng.uniform(0.2, 0.85, rng.integers(2, 4))):
            for _ in range(30):
                r = rng.uniform(*args.radius)
                c = start + f * route + rng.choice([-1, 1]) * rng.uniform(0.05, 0.4) * side
                if ok(c, r):
                    placed.append((c, r))
                    break
        if not walled and len(placed) < 2:  # a wall already blocks the route by itself
            continue
        # 3. Crowded parking spot: an obstacle beside the spot (not on the approach, not between it
        # and the target)
        if want_crowded:
            for _ in range(50):
                r = rng.uniform(*args.radius)
                a = bearing + rng.choice([-1, 1]) * rng.uniform(np.pi / 3, 2 * np.pi / 3)
                c = park + (r + ROBOT_RADIUS + rng.uniform(0.1, 0.3)) * np.array(
                    [np.cos(a), np.sin(a)]
                )
                if ok(c, r, room=0.1):
                    placed.append((c, r))
                    break
            else:
                continue
        # 4. Extras near the route (within 1.2 m of it), not scattered far away
        wanted = min(args.max_obstacles, len(placed) + rng.integers(0, 3))
        for _ in range(200):
            if len(placed) >= wanted:
                break
            r = rng.uniform(*args.radius)
            c = start + rng.uniform(-0.1, 1.1) * route + rng.uniform(-1.2, 1.2) * side
            if ok(c, r):
                placed.append((c, r))

        def rnd(v):
            return [round(float(x), 4) for x in v]

        return {
            "start": rnd([*start, yaw]),
            "target": rnd([*target_xy, z]),
            "park": rnd(park),
            "obstacles": [{"xy": rnd(c), "r": round(float(r), 3)} for c, r in placed],
        }


ROBOT_SPEED = (
    0.2  # m/s: rough average speed of the base, used to time when a mover should cross the route
)


def path_points(c0, d, span, step=0.25):
    """Points along the segment a mover sweeps: it oscillates between c0 - span*d and c0 +
    span*d."""
    return c0 + np.arange(-span, span + 1e-9, step)[:, None] * d


def sample_moving_scenario(rng, args):
    """Start, target and parking spot like sample_scenario; then static obstacles (--static-count)
    that satisfy fits(), then movers (--moving-count). A mover is placed backwards from where it
    should cross the start-park line:
    it crosses at fraction f of the route when the robot is expected to be there, so its start
    is that crossing point minus speed*time along its direction. All points of its whole sweep
    must stay clear of the parking spot, the target and the park-target line, of the static
    obstacles and of the other movers' sweeps."""
    while True:
        start = rng.uniform(-1, 1, size=2)
        yaw = rng.uniform(-np.pi, np.pi)
        bearing = rng.uniform(-np.pi, np.pi)
        ahead = np.array([np.cos(bearing), np.sin(bearing)])
        side = np.array([-np.sin(bearing), np.cos(bearing)])
        target_xy = start + rng.uniform(*args.distance) * ahead
        z = rng.uniform(*args.height)
        park = target_xy - standoff(z) * ahead
        route = park - start
        length = np.linalg.norm(route)
        placed = []
        for _ in range(rng.integers(args.static_count[0], args.static_count[1] + 1)):
            for _ in range(100):
                r = rng.uniform(*args.radius)
                c = start + rng.uniform(0.1, 1.0) * route + rng.uniform(-1.0, 1.0) * side
                if fits(c, r, start, park, target_xy, placed):
                    placed.append((c, r))
                    break
        movers = []
        for _ in range(rng.integers(args.moving_count[0], args.moving_count[1] + 1)):
            for _ in range(100):
                r, speed = rng.uniform(*args.radius), rng.uniform(*args.speed)
                tilt = np.radians(rng.uniform(-30, 30))
                d = rng.choice([-1, 1]) * (np.cos(tilt) * side + np.sin(tilt) * ahead)
                f = rng.uniform(0.25, 0.8)
                t_cross = max(
                    3.0, f * length / ROBOT_SPEED + 3.0 + rng.uniform(-3, 3)
                )  # 3 s: the first turn on the spot
                c0 = start + f * route - d * speed * t_cross
                span = speed * t_cross + rng.uniform(1.0, 2.5)
                pts = path_points(c0, d, span)
                if np.linalg.norm(c0 - start) < r + ROBOT_RADIUS + 0.45:
                    continue
                if np.linalg.norm(pts - park, axis=1).min() < r + ROBOT_RADIUS + 0.35:
                    continue
                if np.linalg.norm(pts - target_xy, axis=1).min() < r + 0.3:
                    continue
                if min(segment_distance(p, park, target_xy) for p in pts) < r + 0.15:
                    continue
                if any(np.linalg.norm(pts - c, axis=1).min() < r + rs + 0.3 for c, rs in placed):
                    continue
                if any(
                    np.linalg.norm(pts[:, None] - m["pts"][None], axis=2).min() < r + m["r"] + 0.5
                    for m in movers
                ):
                    continue
                movers.append({"xy": c0, "r": r, "vel": d * speed, "span": span, "pts": pts})
                break
        if len(movers) < args.moving_count[0]:
            continue

        def rnd(v):
            return [round(float(x), 4) for x in v]

        obstacles = [{"xy": rnd(c), "r": round(float(r), 3)} for c, r in placed]
        obstacles += [
            {
                "xy": rnd(m["xy"]),
                "r": round(float(m["r"]), 3),
                "vel": rnd(m["vel"]),
                "span": round(float(m["span"]), 3),
            }
            for m in movers
        ]
        return {
            "start": rnd([*start, yaw]),
            "target": rnd([*target_xy, z]),
            "park": rnd(park),
            "obstacles": obstacles,
        }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--train", type=int, default=60)
    parser.add_argument("--validation", type=int, default=10)
    parser.add_argument("--test", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--distance", type=float, nargs=2, default=[2.8, 4.0], help="start to target, metres"
    )
    parser.add_argument(
        "--height", type=float, nargs=2, default=[0.35, 0.8], help="target height, metres"
    )
    parser.add_argument(
        "--radius", type=float, nargs=2, default=[0.15, 0.3], help="obstacle radius, metres"
    )
    parser.add_argument("--max-obstacles", type=int, default=3)
    parser.add_argument(
        "--prefix", default="", help="added to split names in ids, e.g. 'b' -> trainb_000"
    )
    parser.add_argument(
        "--moving-count",
        type=int,
        nargs=2,
        default=[0, 0],
        metavar=("LO", "HI"),
        help=(
            "moving obstacles per scenario, uniform in LO..HI (not 0 0 -> sample_moving_scenario)"
        ),
    )
    parser.add_argument(
        "--static-count",
        type=int,
        nargs=2,
        default=[0, 0],
        metavar=("LO", "HI"),
        help="static obstacles per scenario with --moving-count, uniform in LO..HI",
    )
    parser.add_argument(
        "--speed", type=float, nargs=2, default=[0.1, 0.3], help="mover speed, m/s"
    )
    parser.add_argument(
        "--hard", action="store_true", help="hard layouts on purpose (sample_hard_scenario)"
    )
    parser.add_argument(
        "--arm-noise",
        type=float,
        default=0.0,
        help="random arm start pose: each joint uniform within +-this (rad) of the usual start",
    )
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Choose a new scenarios file: {out}")
    rng = np.random.default_rng(args.seed)
    scenarios = []
    for split, count in [
        ("train", args.train),
        ("validation", args.validation),
        ("test", args.test),
    ]:
        for i in range(count):
            sample = (
                sample_moving_scenario
                if args.moving_count[1] > 0
                else sample_hard_scenario
                if args.hard
                else sample_scenario
            )
            scenario = {
                "id": f"{split}{args.prefix}_{i:03d}",
                "split": split,
                **sample(rng, args),
                "seed": int(rng.integers(1_000_000)),
            }
            if args.arm_noise > 0:
                q0 = ARM_START + rng.uniform(-args.arm_noise, args.arm_noise, 7)
                scenario["arm_q0"] = [round(float(a), 4) for a in q0]
            scenarios.append(scenario)
    out.write_text(json.dumps(scenarios, indent=1))
    print(f"Wrote {len(scenarios)} scenarios to {out}")


if __name__ == "__main__":
    main()
