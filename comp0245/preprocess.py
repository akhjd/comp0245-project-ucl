"""Lidar preprocessing for the wheels MLP: 36 raw distances -> features that mean something.

The same Preprocessor runs on the saved expert and DAgger runs (training) and on the live lidar
(driving), so the two cannot disagree. It uses only the robot's own lidar, its pose and the dots'
positions relative to it, never the scenario file. It keeps a little memory (object tracks,
remembered doorways), so feed it every scan of a run in time order, and reset() it between runs.

    pre = Preprocessor()
    features = pre.step(t, lidar, (x, y, yaw), dots)      # dots: (3, 4) forward, left, up,
    still-there flag, robot frame

Steps:
  1. Rays to points around the robot; neighbouring hits on one surface form a cluster.
  2. Walls: each cluster is split into straight pieces (split-and-merge); long pieces count as
     walls.
  3. Openings: a gap between the end of one surface and the start of another, with the lidar seeing
     further through it, 0.9-2.6 m wide. Doorways are 1.4 m; the ray spacing makes them look up to
     about 1 m wider. An opening is door-like if the surfaces on both sides are straight and in
     line with it. Gaps between furniture, cylinders and walls are openings too: the robot can pass
     them.
  4. Door memory (feature set v2): every door-like opening is stored in world coordinates using the
     pose. Sightings within 0.6 m of a stored doorway are averaged into it; one seen in 3 or more
     scans is confirmed. The MLP gets the nearest confirmed doorway and, for each dot, the doorway
     in a wall between the robot and the dot (if any), even after the robot has turned away from
     it. Only the robot's own lidar and pose are used; it is reset at the start of every run.
  5. Line to each dot: blocked if the lidar hits something nearer than the dot in that direction.
  6. Objects: small clusters standing in front of what is behind them, tracked in world
     coordinates using the pose, so the robot's own motion cancels out. A track that moves
     faster than 0.08 m/s over 0.8-1 s counts as moving. The robot's velocity is never an input.
"""

import collections

import numpy as np

RAYS, RANGE = 36, 4.0  # as obstacle_env: ray 0 straight ahead, counter-clockwise, 10 degrees apart
ANGLES = np.arange(RAYS) * 2 * np.pi / RAYS
DIRECTIONS = np.stack([np.cos(ANGLES), np.sin(ANGLES)], axis=1)
HIT = RANGE - 1e-6  # a ray at full range hit nothing
MAX_DOTS = 3
BREAK_M, BREAK_SHARE = (
    0.4,
    0.3,
)  # neighbouring hits further apart than max(0.4 m, 0.3 x range) are separate surfaces
SPLIT_M = 0.08  # a piece is straight if no point is further than this from its end-to-end line
WALL_MIN_M = 0.8  # a straight piece at least this long, with 3 or more points, is a wall
OBJECT_MAX_M = 0.8  # a cluster no wider than this, with no wall in it, is an object
OPENING_M = (0.9, 2.6)  # accepted opening widths
OPENING_DEPTH_M = 0.4  # rays through an opening must reach this much further than both edges
OPENING_MAX_RAYS = 12  # an opening spans at most 120 degrees
OBJECT_CENTRE_M = (
    0.12  # the lidar sees an object's near side: its centre is this much further away
)
TRACK_GATE_M, TRACK_LOST_S, TRACK_WINDOW_S = 0.5, 0.5, 1.0
TRACK_EVERY_S = 0.05  # a track keeps one position per 50 ms, enough for its velocity
MOVING_M_S, MOVING_MIN_S = 0.08, 0.8  # moving: faster than this over a history at least this long
ISOLATED_M = (
    0.3  # an object stands in front: the rays either side reach at least this much further
)
DOOR_PARALLEL = np.cos(
    np.radians(20)
)  # an opening is door-like if its two edges and the gap all lie on one line
BLOCK_MARGIN_M = (
    0.5  # the line to a dot is blocked by something at least this much nearer than the dot
)
HORIZON_S = 5.0  # longest time to closest approach reported
DOOR_MATCH_M = 0.6  # a door-like sighting this close to a remembered doorway is the same doorway
DOOR_CONFIRM = 3  # a remembered doorway is used once seen in this many scans...
DOOR_WALL_M = (
    1.0  # ...and once, in some scan, a straight piece this long was seen beside it: furniture
)
# sides that happen to line up look door-like too, but are shorter than walls
DOOR_KEEP = 6  # at most this many remembered (rooms have 1-2 doorways); the least seen go first
DOOR_AGE_MAX_S = 30.0  # time since a doorway was last seen is reported up to this

MEMORY_NAMES = [
    "doors_remembered",
    "door_seen",
    "door_distance",
    "door_sin",
    "door_cos",
    "door_dir_sin2",
    "door_dir_cos2",
    "door_age_s",
] + [
    f"dot{d}_door_{n}"
    for d in range(MAX_DOTS)
    for n in ("seen", "distance", "sin", "cos", "across")
]
SECTOR_CHANGE_S = 0.5  # sector change: nearest distance now minus this long ago
NAMES = (
    [f"sector{30 * j}deg_nearest" for j in range(12)]
    + [f"sector{30 * j}deg_change" for j in range(12)]
    + ["closest_distance", "closest_sin", "closest_cos"]
    + ["wall_seen", "wall_distance", "wall_sin", "wall_cos", "wall_dir_sin2", "wall_dir_cos2"]
    + [
        "openings",
        "opening_seen",
        "opening_distance",
        "opening_sin",
        "opening_cos",
        "opening_width",
        "opening_door_like",
    ]
    + [
        f"dot{d}_{n}"
        for d in range(MAX_DOTS)
        for n in (
            "blocked",
            "free",
            "opening_seen",
            "opening_distance",
            "opening_sin",
            "opening_cos",
            "opening_door_like",
        )
    ]
    + [
        "mover_seen",
        "mover_distance",
        "mover_sin",
        "mover_cos",
        "mover_v_forward",
        "mover_v_left",
        "mover_closest_m",
        "mover_closest_s",
    ]
    + ["object_seen", "object_distance", "object_sin", "object_cos"]
    + MEMORY_NAMES
)


TRACKED = tuple(n for n in NAMES if n.startswith(("mover_", "object_")))
MEMORY = tuple(MEMORY_NAMES)
# The MLP's set (chosen 3 Nov): mover tracking is unreliable among furniture
# (tests/check_preprocess.py), so it is left out;
# movers still show in the sector changes.
MLP_NAMES = [n for n in NAMES if n not in TRACKED and n not in MEMORY]
VERSION = "v1"  # change when the features change, so old caches are not reused
# Feature sets by cache version. v1: pre1. v2: v1 plus the door memory
# (comp0245/training/train_mlp.py --door-memory).
FEATURE_SETS = {"v1": MLP_NAMES, "v2": MLP_NAMES + list(MEMORY)}


def clusters(r):
    """Neighbouring rays that hit the same surface, in angular order. Returns (list of index
    arrays, points (36, 2))."""
    hit = r < HIT
    pts = r[:, None] * DIRECTIONS
    after = np.roll(np.arange(RAYS), -1)
    gap = np.linalg.norm(pts - pts[after], axis=1)
    joined = (
        hit & hit[after] & (gap <= np.maximum(BREAK_M, BREAK_SHARE * np.minimum(r, r[after])))
    )  # ray k joined to k+1
    if not hit.any():
        return [], pts
    if joined.all():
        return [np.arange(RAYS)], pts
    start = (int(np.argmin(joined)) + 1) % RAYS  # the ray after a break
    groups, current = [], []
    for i in range(RAYS):
        k = (start + i) % RAYS
        if hit[k]:
            current.append(k)
        if not joined[k] and current:
            groups.append(np.array(current))
            current = []
    return groups, pts


def straight_pieces(pts):
    """Split one cluster's points (in order) into straight pieces. Returns [(first point, last
    point, number of points)]."""
    out = []

    def split(i, j):
        if j - i >= 2:
            a, b = pts[i], pts[j]
            ab, rel = b - a, pts[i + 1 : j] - a
            off = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / max(np.linalg.norm(ab), 1e-9)
            k = int(np.argmax(off))
            if off[k] > SPLIT_M:
                split(i, i + 1 + k)
                split(i + 1 + k, j)
                return
        out.append((pts[i], pts[j], j - i + 1))

    split(0, len(pts) - 1)
    return out


def closest_on_segment(a, b):
    """Point of segment a-b closest to the robot (the origin)."""
    ab = b - a
    s = np.clip(-a @ ab / max(ab @ ab, 1e-12), 0.0, 1.0)
    return a + s * ab


def edge_direction(pts, g, end):
    """Unit direction of the straight piece at one end of a cluster, or None if that piece has
    fewer than 2 points."""
    pieces = straight_pieces(pts[g])
    a, b, count = pieces[-1] if end else pieces[0]
    d = b - a
    return d / np.linalg.norm(d) if count >= 2 and np.linalg.norm(d) > 1e-6 else None


def door_like(pts, ga, gb, a, b):
    """True if the surfaces on both sides of the gap a-b are straight and in line with the gap,
    like a wall with a doorway."""
    gap = (pts[b] - pts[a]) / max(np.linalg.norm(pts[b] - pts[a]), 1e-9)
    da, db = edge_direction(pts, ga, True), edge_direction(pts, gb, False)
    return (
        da is not None
        and db is not None
        and abs(da @ gap) > DOOR_PARALLEL
        and abs(db @ gap) > DOOR_PARALLEL
    )


def openings(groups, r, pts):
    """Gaps between the end of one surface and the start of a later one that the lidar sees
    through. Returns [(middle point, width, door-like, unit direction across the gap, longer
    straight piece beside it)] in the robot frame."""
    found, n = [], len(groups)
    for g in range(n):
        a = groups[g][-1]
        for step in range(1, n):
            b = groups[(g + step) % n][0]
            span = (b - a) % RAYS
            if span > OPENING_MAX_RAYS:
                break
            between = (a + np.arange(1, span)) % RAYS
            if len(between) == 0:
                continue  # touching surfaces: look one further
            if not np.all(r[between] > max(r[a], r[b]) + OPENING_DEPTH_M):
                break  # something nearer in between: no opening here
            width = float(np.linalg.norm(pts[a] - pts[b]))
            if OPENING_M[0] <= width <= OPENING_M[1]:
                gb = groups[(g + step) % n]
                side = max(
                    np.linalg.norm(np.subtract(*straight_pieces(pts[groups[g]])[-1][:2])),
                    np.linalg.norm(np.subtract(*straight_pieces(pts[gb])[0][:2])),
                )
                found.append(
                    (
                        (pts[a] + pts[b]) / 2,
                        width,
                        door_like(pts, groups[g], gb, a, b),
                        (pts[b] - pts[a]) / width,
                        side,
                    )
                )
            break
    return found


def polar(p):
    d = float(np.hypot(*p))
    return d, (p[1] / d if d > 1e-9 else 0.0), (p[0] / d if d > 1e-9 else 1.0)


class Track:
    def __init__(self, t, xy):
        self.history = collections.deque(
            [(t, xy)]
        )  # one position per TRACK_EVERY_S, over the last TRACK_WINDOW_S
        self.last = (t, xy)
        self.seen_now, self.v = True, None

    def add(self, t, xy):
        self.last = (t, xy)
        if t - self.history[-1][0] >= TRACK_EVERY_S - 1e-9:
            self.history.append((t, xy))
            while self.history[-1][0] - self.history[0][0] > TRACK_WINDOW_S:
                self.history.popleft()
            self.v = self.velocity()

    def velocity(self):
        """Least-squares velocity over the history, or None if it is too short to judge."""
        t = np.array([h[0] for h in self.history])
        if t[-1] - t[0] < MOVING_MIN_S:
            return None
        xy = np.array([h[1] for h in self.history])
        tc = t - t.mean()
        return (tc @ (xy - xy.mean(axis=0))) / (tc @ tc)

    def predicted(self, t):
        return self.last[1] + (self.v if self.v is not None else 0.0) * (t - self.last[0])


class Preprocessor:
    def __init__(self, names=None):
        """names: which features step() returns, in this order (default all). Tracking runs only
        if a tracked one is asked for."""
        self.names = list(NAMES if names is None else names)
        self.columns = np.array([NAMES.index(n) for n in self.names])
        self.tracking = any(n in TRACKED for n in self.names)
        self.memory = any(n in MEMORY for n in self.names)
        self.reset()

    def reset(self):
        self.tracks = []
        self.doors = []  # remembered doorways, world frame
        self.sector_history = collections.deque()

    def step(self, t, lidar, pose, dots):
        r = np.clip(np.asarray(lidar, dtype=float), 0.0, RANGE)
        x, y, yaw = pose
        c, s = np.cos(yaw), np.sin(yaw)

        def to_world(p):
            return np.array([x + c * p[0] - s * p[1], y + s * p[0] + c * p[1]])

        def to_robot(v):
            return np.array(
                [c * v[0] + s * v[1], -s * v[0] + c * v[1]]
            )  # a world direction seen from the robot

        out = []
        sectors = np.roll(r, 1).reshape(12, 3).min(axis=1)  # sector j covers rays 3j-1, 3j, 3j+1
        self.sector_history.append((t, sectors))
        while (
            len(self.sector_history) > 1
            and self.sector_history[1][0] <= t - SECTOR_CHANGE_S + 1e-9
        ):
            self.sector_history.popleft()  # [0] is now the scan 0.5 s ago (or the run's first)
        out += list(sectors) + list(sectors - self.sector_history[0][1])
        k = int(np.argmin(r))
        out += [r[k], np.sin(ANGLES[k]), np.cos(ANGLES[k])]

        groups, pts = clusters(r)
        walls, objects = [], []
        for g in groups:
            pieces = straight_pieces(pts[g])
            long = [
                (a, b)
                for a, b, count in pieces
                if count >= 3 and np.linalg.norm(b - a) >= WALL_MIN_M
            ]
            walls += long
            before, after = (g[0] - 1) % RAYS, (g[-1] + 1) % RAYS
            isolated = r[before] > r[g[0]] + ISOLATED_M and r[after] > r[g[-1]] + ISOLATED_M
            if (
                not long
                and isolated
                and len(g) < RAYS
                and np.linalg.norm(pts[g[0]] - pts[g[-1]]) <= OBJECT_MAX_M
            ):
                centre = pts[g].mean(axis=0)
                objects.append(
                    centre + OBJECT_CENTRE_M * centre / max(np.linalg.norm(centre), 1e-9)
                )
        if walls:
            near = [closest_on_segment(a, b) for a, b in walls]
            w = int(np.argmin([np.linalg.norm(p) for p in near]))
            d, si, co = polar(near[w])
            a, b = walls[w]
            angle = np.arctan2(*(b - a)[::-1])
            out += [1.0, d, si, co, np.sin(2 * angle), np.cos(2 * angle)]
        else:
            out += [0.0, RANGE, 0.0, 1.0, 0.0, 1.0]

        gaps = openings(groups, r, pts)
        if gaps:
            middle, width, door, *_ = min(gaps, key=lambda o: np.linalg.norm(o[0]))
            out += [min(len(gaps), 3), 1.0, *polar(middle), width, float(door)]
        else:
            out += [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

        for forward, left, _, there in np.asarray(dots, dtype=float).reshape(MAX_DOTS, 4):
            if there < 0.5:
                out += [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                continue
            heading, distance = np.arctan2(left, forward), np.hypot(forward, left)
            ray = heading / (2 * np.pi / RAYS)
            free = min(r[int(np.floor(ray)) % RAYS], r[int(np.ceil(ray)) % RAYS])
            blocked = float(free < min(distance, HIT) - BLOCK_MARGIN_M)
            if gaps:
                middle, _, door, *_ = min(
                    gaps,
                    key=lambda o: abs(
                        np.angle(np.exp(1j * (np.arctan2(o[0][1], o[0][0]) - heading)))
                    ),
                )
                out += [blocked, free, 1.0, *polar(middle), float(door)]
            else:
                out += [blocked, free, 0.0, 0.0, 0.0, 1.0, 0.0]

        if self.tracking:
            out += self.track(t, [to_world(p) for p in objects], to_robot, np.array([x, y]))
        else:
            out += [0.0] * len(TRACKED)
        if self.memory:
            out += self.remember(
                t,
                [
                    (to_world(m), np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]]), side)
                    for m, _, door, u, side in gaps
                    if door
                ],
                to_robot,
                np.array([x, y]),
                dots,
            )
        else:
            out += [0.0] * len(MEMORY)
        return np.array(out, dtype=np.float32)[self.columns]

    def remember(self, t, sightings, to_robot, robot, dots):
        """Add this scan's door-like openings (world middle, world direction across the gap,
        longer side) to the remembered doorways, then describe the confirmed ones from where the
        robot is now."""
        for middle, u, side in sightings:
            double = np.array(
                [u[0] ** 2 - u[1] ** 2, 2 * u[0] * u[1]]
            )  # direction doubled: a gap seen from either side agrees
            near = [d for d in self.doors if np.linalg.norm(d["xy"] - middle) < DOOR_MATCH_M]
            if near:
                d = min(near, key=lambda d: np.linalg.norm(d["xy"] - middle))
                d["xy"] = (d["xy"] * d["count"] + middle) / (d["count"] + 1)
                d["double"] = d["double"] + double
                d["count"] += 1
                d["last"] = t
                d["side"] = max(d["side"], side)
            else:
                self.doors.append(
                    {"xy": middle, "double": double, "count": 1, "last": t, "side": side}
                )
        if len(self.doors) > DOOR_KEEP:
            self.doors = sorted(self.doors, key=lambda d: (d["count"], d["last"]), reverse=True)[
                :DOOR_KEEP
            ]
        confirmed = []
        for d in self.doors:
            if confirmed_door(d):
                angle = (
                    np.arctan2(d["double"][1], d["double"][0]) / 2
                )  # wall direction, world frame
                along = to_robot(
                    np.array([np.cos(angle), np.sin(angle)])
                )  # ...seen from the robot
                confirmed.append((to_robot(d["xy"] - robot), along, t - d["last"]))
        out = [float(min(len(confirmed), 3))]
        if confirmed:
            p, along, age = min(confirmed, key=lambda c: np.linalg.norm(c[0]))
            angle = np.arctan2(along[1], along[0])
            out += [1.0, *polar(p), np.sin(2 * angle), np.cos(2 * angle), min(age, DOOR_AGE_MAX_S)]
        else:
            out += [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]

        def cross(a, b):
            return a[0] * b[1] - a[1] * b[0]

        for forward, left, _, there in np.asarray(dots, dtype=float).reshape(MAX_DOTS, 4):
            dot = np.array([forward, left])
            if there < 0.5 or not confirmed:
                out += [0.0, 0.0, 0.0, 1.0, 0.0]
                continue
            # The doorway whose wall line has the robot on one side and the dot on the other; if
            # several (or none), the
            # one with the shortest way robot -> doorway -> dot
            options = [
                (
                    cross(along, -p) * cross(along, dot - p) < 0,
                    np.linalg.norm(p) + np.linalg.norm(dot - p),
                    p,
                )
                for p, along, _ in confirmed
            ]
            across, _, p = min(options, key=lambda o: (not o[0], o[1]))
            out += [1.0, *polar(p), float(across)]
        return out

    def track(self, t, seen, to_robot, robot):
        """Match this scan's objects to tracks (nearest first, within the gate), then describe
        the nearest moving one and the nearest still one."""
        for tr in self.tracks:
            tr.seen_now = False
        pairs = []
        if self.tracks and seen:
            gaps = np.linalg.norm(
                np.array([tr.predicted(t) for tr in self.tracks])[:, None] - np.array(seen)[None],
                axis=2,
            )
            pairs = sorted(
                (gaps[i, j], i, j) for i, j in zip(*np.nonzero(gaps <= TRACK_GATE_M), strict=False)
            )
        used_t, used_o = set(), set()
        for gap, i, j in pairs:
            if gap > TRACK_GATE_M or i in used_t or j in used_o:
                continue
            used_t.add(i)
            used_o.add(j)
            tr = self.tracks[i]
            tr.add(t, seen[j])
            tr.seen_now = True
        self.tracks = [tr for tr in self.tracks if t - tr.last[0] <= TRACK_LOST_S]
        self.tracks += [Track(t, p) for j, p in enumerate(seen) if j not in used_o]
        movers, still = [], []
        for tr in self.tracks:
            if not tr.seen_now:
                continue
            v = tr.v
            (movers if v is not None and np.linalg.norm(v) > MOVING_M_S else still).append((tr, v))
        out = []
        if movers:
            tr, v = min(movers, key=lambda m: np.linalg.norm(m[0].last[1] - robot))
            p, u = to_robot(tr.last[1] - robot), to_robot(v)
            when = float(np.clip(-(p @ u) / max(u @ u, 1e-9), 0.0, HORIZON_S))
            out += [1.0, *polar(p), u[0], u[1], float(np.linalg.norm(p + u * when)), when]
        else:
            out += [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        if still:
            tr, _ = min(still, key=lambda m: np.linalg.norm(m[0].last[1] - robot))
            out += [1.0, *polar(to_robot(tr.last[1] - robot))]
        else:
            out += [0.0, 0.0, 0.0, 1.0]
        return out


def confirmed_door(d):
    """A remembered doorway the MLP is told about."""
    return d["count"] >= DOOR_CONFIRM and d["side"] >= DOOR_WALL_M


def pose_of(observation):
    """(x, y, yaw) from one saved or live observation row (x, y at 0, 1; sin, cos yaw at 3, 4)."""
    return observation[0], observation[1], float(np.arctan2(observation[3], observation[4]))


def process_run(time_s, lidar, observations, dots, names=None):
    """Features for every row of one run, in time order. dots: (rows, 3, 4) forward, left, up,
    still-there. Returns (rows, len(names))."""
    pre = Preprocessor(names)
    return np.stack(
        [
            pre.step(t, scan, pose_of(o), d)
            for t, scan, o, d in zip(time_s, lidar, observations, dots, strict=False)
        ]
    )


def cached_features(folder, time_s, lidar, observations, dots, version=VERSION):
    """The MLP's features (set `version` of FEATURE_SETS) for one saved run, from its cache file
    if there is one, else computed and saved."""
    from pathlib import Path

    names = FEATURE_SETS[version]
    path = Path(folder) / f"features_{version}.npy"
    if path.exists():
        f = np.load(path)
        if f.shape == (len(time_s), len(names)):
            return f
    f = process_run(time_s, lidar, observations, dots, names)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, f)
    tmp.replace(path)  # never leave a half-written cache
    return f
