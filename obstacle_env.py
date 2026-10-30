"""Obstacle world: static cylinders, a random start pose, a 2D lidar and a steer-around expert for the base.

Builds on robot_env.Robot (unchanged) and the course arm expert. The base expert is ours:
a memoryless "steer-around" (vortex) field, so its wheel command depends only on what it senses right now,
never on an internal mode. It is sensor-only: lidar, own pose/velocity and the target position, exactly what
a learned policy gets. Obstacle positions are used only to build the world and to count contacts (referee).
Exception: RRTExpert deliberately reads the true obstacles (teacher-student learning).
Physics change for obstacle runs only: wheel sideways grip 0.1 instead of the course's 0.01 (see ObstacleRobot).
"""
import collections
import contextlib
import json
import time
from pathlib import Path
import numpy as np
from models import clip_action
from reach_zone import parking_distance
from robot_env import Robot, provenance

ROBOT_RADIUS = 0.47        # m: circle around the base footprint (wheel boxes reach x 0.342, y 0.313)
OBSTACLE_HEIGHT = 0.6      # m: taller than the lidar plane
LIDAR_RAYS = 36            # one ray every 10 degrees, ray 0 = straight ahead, counter-clockwise
LIDAR_START = 0.5          # m from base centre: rays start just outside the robot body
LIDAR_RANGE = 4.0          # m from base centre
LIDAR_HEIGHT = 0.3         # m above the base frame
LIDAR_ANGLES = np.arange(LIDAR_RAYS) * 2*np.pi/LIDAR_RAYS
WHEEL_SIDE_GRIP = 0.1      # sideways wheel friction factor; the course uses 0.01 (see ObstacleRobot)
NAV_TIMEOUT = 50.0         # s: detours with stop-turn-go take longer than the course's 25 s open-ground limit

WALL_HEIGHT, WALL_THICKNESS = 1.0, 0.1   # m: room walls, taller than the lidar plane

def fit_to_room(scenario, half):
    """Copy of a scenario inside a square room with walls at +-half metres ("room": [xmin, ymin, xmax, ymax]).
    The robot's start, parking spot and the target must fit (ValueError otherwise). Obstacles that do not fit are
    dropped. A mover keeps its start and velocity, but its sweep is cut at the walls: "sweep": [lo, hi] are the
    signed distances from its start along its first direction at which it turns round. Returns (scenario, dropped)."""
    lo, hi = np.array([-half, -half], dtype=float), np.array([half, half], dtype=float)
    inside = lambda xy, m: bool(np.all(np.asarray(xy) >= lo + m) and np.all(np.asarray(xy) <= hi - m))
    if not (inside(scenario["start"][:2], ROBOT_RADIUS + 0.05) and inside(scenario["park"], ROBOT_RADIUS + 0.05)
            and inside(scenario["target"][:2], 0.05)):
        raise ValueError(f"{scenario['id']}: start, parking spot or target outside a room of half-size {half}")
    kept, dropped = [], 0
    for o in scenario["obstacles"]:
        if not inside(o["xy"], o["r"]):
            dropped += 1
            continue
        if "vel" in o and np.any(np.asarray(o["vel"]) != 0):
            c, d = np.asarray(o["xy"], dtype=float), np.asarray(o["vel"], dtype=float)
            d = d/np.linalg.norm(d)
            up, down = float(o.get("span", np.inf)), float(o.get("span", np.inf))
            for axis in range(2):
                if abs(d[axis]) > 1e-9:
                    far = (hi[axis] - o["r"] - c[axis])/abs(d[axis])
                    near = (c[axis] - lo[axis] - o["r"])/abs(d[axis])
                    up, down = min(up, far if d[axis] > 0 else near), min(down, near if d[axis] > 0 else far)
            o = {**o, "sweep": [-down, up]}
        kept.append(o)
    return {**scenario, "obstacles": kept, "room": [*lo.tolist(), *hi.tolist()]}, dropped

def wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def standoff(target_z):
    """How far the base parks from the target (m): Akhilesh's reach zone (reach_zone.py), 75% of the arm's
    reach circle at the target's height. Replaces the course rule interp(z, [0.1, 0.5, 1.1, 1.25],
    [0.80, 0.65, 0.65, 0.35]) used for archive/runs/obst-v1."""
    return float(parking_distance(target_z))

def lidar_points(p, yaw, lidar):
    """Lidar hits as world-frame points, grouped into clusters of neighbouring rays (one cluster per object)."""
    hit = lidar < LIDAR_RANGE - 1e-6
    pts = p + lidar[:, None] * np.stack([np.cos(yaw + LIDAR_ANGLES), np.sin(yaw + LIDAR_ANGLES)], axis=1)
    clusters, current = [], []
    for k in range(LIDAR_RAYS):
        if hit[k] and current and abs(lidar[k] - lidar[current[-1]]) < 0.3:
            current.append(k)
        else:
            if current:
                clusters.append(current)
            current = [k] if hit[k] else []
    if current:
        if clusters and clusters[0][0] == 0 and hit[-1] and abs(lidar[0] - lidar[-1]) < 0.3:
            clusters[0] = current + clusters[0]      # the ring wraps around: ray 35 neighbours ray 0
        else:
            clusters.append(current)
    return [pts[c] for c in clusters], [lidar[c] for c in clusters]

class SteerAround:
    """Sensor-only, memoryless base expert: pull toward the parking spot + slide around what the lidar sees.

    Inputs are only what the learned policy also gets: the base pose and velocity, the 36 lidar ranges and the
    target position. The parking spot is recomputed every step as the point at standoff distance from the
    target on the line from the robot, so no start pose or map is needed.
    Every lidar cluster (one seen object) closer than `influence` (gap between the robot circle and its nearest
    hit) adds a tangential push, on the side away from the object's centre of hits, and a push away. The blend
    gives a desired heading: big errors -> stop, then spin slowly; otherwise drive with gentle curves.
    Measured base behaviour: spinning on the spot slides the base about 0.3 m per half turn, so spins are slow
    and only for big heading errors.
    """
    influence = 1.0        # m of gap where obstacles start to matter
    near_goal = 0.3        # m: influence shrinks to the distance left to the goal, but not below this
    too_close = 0.15       # m of gap: below this, also push straight away
    # Spinning on the spot moves the base origin, so the "at the spot, now face the target" zone must be wider
    # than that or turning pushes the base back out and it drives again.
    pos_tol, ang_tol = 0.15, 0.05
    park_tol = 0.05        # m: inside pos_tol, creep along the robot-target line until this close to the spot
    creep_v = 0.1          # m/s: top speed of that creep (the base brakes in under 0.25 s)
    max_v, min_v, kp_v = 0.25, 0.05, 0.6
    spin_tol = 1.2         # rad: heading errors larger than this mean brake, then turn on the spot
    max_spin, max_curve = 0.8, 0.6   # rad/s: spinning on the spot / steering while driving
    stopped = 0.05         # m/s of forward speed: slow enough to turn on the spot
    reverse_zone = 0.3     # m: this close to the goal, back up instead of turning round if it is behind
    tie_zone = 0.35        # sin(20 deg): objects this close to the goal line are passed on the side faced
    wheel_radius, track = 0.11, 0.66   # same skid-steer numbers as the course expert

    def direction(self, p, yaw, lidar, goal):
        """Desired unit direction of travel (world frame), plus the strongest obstacle weight.

        For each object the lidar sees, only the part of the goal direction that heads INTO the object is
        turned to slide past it, by a weight that grows from 0 (gap = influence) to 1 (touching). Far away this
        is a small nudge; at the object it becomes a pure sideways slide, so the robot never swings wide and
        never closes in head-on. Near the goal the influence shrinks, so objects beside the parking spot do not
        push the robot off it.
        """
        to_goal = goal - p
        dist = np.linalg.norm(to_goal)
        a = to_goal / max(dist, 1e-9)
        influence = float(np.clip(dist, self.near_goal, self.influence))
        d = a.copy()
        w_max = 0.0
        for pts, ranges in zip(*lidar_points(p, yaw, lidar)):
            nearest = pts[np.argmin(ranges)]
            gap = ranges.min() - ROBOT_RADIUS
            if gap >= influence:
                continue
            away = (p - nearest) / max(np.linalg.norm(p - nearest), 1e-9)
            w = float(np.clip(1 - gap/influence, 0, 1))
            # How much the goal direction, or the current heading (still turning toward the goal), heads into it
            into = max(0.0, float(np.dot(a, -away)), float(np.dot([np.cos(yaw), np.sin(yaw)], -away)))
            t = np.array([-away[1], away[0]])  # counter-clockwise tangent
            # t passes the object on the right; -t on the left. Pass on the side away from the object's centre
            # (mean of its hits). Near a tie, pass on the side the robot already faces: with one ray per 10
            # degrees, the apparent centre shifts as the robot turns, and any fixed rule then has a boundary
            # the robot can sit on, flipping sides forever. Preferring the faced side makes turning reinforce
            # the choice instead.
            to_centre = pts.mean(axis=0) - p
            side = (a[0]*to_centre[1] - a[1]*to_centre[0]) / max(np.linalg.norm(to_centre), 1e-9)
            if abs(side) < self.tie_zone:
                if np.dot(t, [np.cos(yaw), np.sin(yaw)]) < 0:
                    t = -t
            elif side < 0:
                t = -t
            d += w*into*(away + t)                     # turn the into-object part sideways
            if gap < self.too_close:                   # safety net: back off if somehow very close
                d += (1 - gap/self.too_close)*away
            w_max = max(w_max, w*into)
        return d / max(np.linalg.norm(d), 1e-9), w_max

    def wheels(self, p, yaw, velocity, lidar, target, goal=None, speed_cap=None):
        """Wheel speeds [FR, FL, BL, BR] in rad/s, and whether the base is parked facing the target.
        goal: steer to this point instead of the parking spot (a waypoint of a planner, at least 0.3 m away);
        speed_cap: upper limit (m/s) on the driving speed."""
        to_target = target[:2] - p
        r = np.linalg.norm(to_target)
        if goal is None:
            goal = target[:2] - standoff(target[2]) * to_target / max(r, 1e-9)
        dist = np.linalg.norm(goal - p)
        # Only forward speed needs braking before a turn; spinning on the spot shows up as sideways speed.
        moving = abs(np.cos(yaw)*velocity[0] + np.sin(yaw)*velocity[1]) > self.stopped
        arrived = False
        if dist < self.pos_tol:
            # The spot is always on the robot-target line, so parking is: face the target, then creep forwards
            # (too far) or backwards (too close) until the distance error is under park_tol.
            err = wrap(np.arctan2(to_target[1], to_target[0]) - yaw)
            radial = r - standoff(target[2])
            if abs(err) > 3*self.ang_tol:
                v, w = 0.0, (0.0 if moving else float(np.clip(3.0*err, -self.max_spin, self.max_spin)))
            elif abs(radial) > self.park_tol:
                v = float(np.clip(self.kp_v*radial, -self.creep_v, self.creep_v))
                w = float(np.clip(3.0*err, -self.max_curve, self.max_curve))
            else:
                arrived = abs(err) < self.ang_tol and not moving
                v, w = 0.0, (0.0 if arrived or moving else float(np.clip(3.0*err, -self.max_spin, self.max_spin)))
        else:
            d, w_obs = self.direction(p, yaw, lidar, goal)
            err = wrap(np.arctan2(d[1], d[0]) - yaw)
            sign = 1.0
            if dist < self.reverse_zone and abs(err) > np.pi/2:   # small overshoot: reverse onto the spot
                sign, err = -1.0, wrap(err - np.pi)
            if abs(err) > self.spin_tol:
                v, w = 0.0, (0.0 if moving else float(np.clip(3.0*err, -self.max_spin, self.max_spin)))
            else:
                v = sign * np.clip(self.kp_v*dist, self.min_v, self.max_v) * np.cos(err) * (1 - 0.5*w_obs)
                if speed_cap is not None:
                    v = float(np.clip(v, -speed_cap, speed_cap))
                w = float(np.clip(1.5*err, -self.max_curve, self.max_curve))
        left = (v - w*self.track/2) / self.wheel_radius
        right = (v + w*self.track/2) / self.wheel_radius
        return np.array([right, left, left, right]), arrived

class RRTExpert(SteerAround):
    """Privileged base expert: knows every obstacle's exact position, radius, velocity and turning points (no lidar)
    and drives along a kinodynamic, time-extended RRT plan to the parking spot.

    A tree node is (x, y, yaw, time). An edge is a short run of one of the base's real motions (`controls`: forward
    arcs, spin on the spot, reverse) or a wait, so plans respect that the base cannot move sideways. Every sample of
    every edge is checked against where each obstacle will be at that time. `connect` tries to finish from a node
    with "turn to the goal, then drive straight", which gives short plans. The earliest arrival found wins, and
    `shorten` then removes detours and needless waits. Waiting is only considered when something is moving.
    The plan is kept while it stays valid, so the route does not change without a reason; it is replaced when the
    robot strays from it or it runs out. Random numbers are seeded from the time, so runs are reproducible.
    A plan is collision-free for the disc model and exact predictions; the real base can still slip off it.
    Parking (last `park_handover` m) is SteerAround's."""
    margins = (0.12, 0.04)         # m of clearance around the robot disc: tried in this order
    horizon_extra = 25.0           # s the plan may take beyond the straight run
    iters, iters_after_found = 1500, 200
    shorten_iters, wait_trim = 100, 0.5   # tries at shortening a plan; s removed from a wait at a time
    wait_choices = (1.0, 2.0, 4.0)
    v_plan, w_plan = 0.2, 0.6      # m/s, rad/s of connect's straight run and spin (below the limits, to keep up)
    sample_dt, edge_steps = 0.1, 5 # s between samples; samples per edge
    max_stray = 0.3                # m from the plan's reference that triggers a new plan
    check_every = 0.25             # s between plan checks
    park_handover, park_exit = 0.3, 0.6
    controls = np.array([[0.2, 0.0], [0.2, 0.25], [0.2, -0.25], [0.2, 0.5], [0.2, -0.5], [0.1, 0.5], [0.1, -0.5],
                         [0.0, 0.6], [0.0, -0.6], [-0.1, 0.0], [-0.1, 0.4], [-0.1, -0.4]])

    def __init__(self, robot):
        self.robot = robot
        self.waiting, self.wait_time, self.last_t, self.now = False, 0.0, None, 0.0
        self.plan, self.next_check, self.parking, self.plans, self.plan_failures = None, 0.0, False, 0, 0
        self.edges = self.rollouts()

    def rollouts(self):
        """Each control's motion from the origin: array (controls, edge_steps, 3) of x, y, yaw."""
        out = np.zeros((len(self.controls), self.edge_steps, 3))
        state = np.zeros((len(self.controls), 3))
        for k in range(self.edge_steps):
            state = state + self.sample_dt*np.stack([self.controls[:, 0]*np.cos(state[:, 2]),
                                                     self.controls[:, 0]*np.sin(state[:, 2]), self.controls[:, 1]], axis=1)
            out[:, k] = state
        return out

    def observe(self, t, p, yaw, lidar):
        if self.waiting and self.last_t is not None:
            self.wait_time += t - self.last_t
        self.now, self.last_t = t, t

    def load_world(self):
        r = self.robot
        speed = np.linalg.norm(r.obstacle_vel0, axis=1)
        self.d0 = r.obstacle_vel0 / np.maximum(speed, 1e-9)[:, None]
        self.start, self.radius, self.speed, self.sweep = r.obstacle_start.copy(), r.obstacle_r, speed, r.obstacle_sweep
        self.room = r.room
        self.s0 = np.sum((r.obstacle_xy - r.obstacle_start)*self.d0, axis=1)
        self.dir = np.sign(np.sum(r.obstacle_vel*self.d0, axis=1))

    def predict(self, ts):
        """Obstacle centres at times ts (relative to now): (len(ts), obstacles, 2). Movers follow the same
        back-and-forth between their sweep limits as ObstacleRobot.advance_obstacles."""
        x = self.s0 + self.dir*self.speed*ts[:, None]
        lo, width = self.sweep[:, 0], self.sweep[:, 1] - self.sweep[:, 0]
        finite = np.isfinite(width) & (width > 1e-9)
        if finite.any():
            lo, W = np.where(finite, lo, 0.0), np.where(finite, width, 1.0)
            y = np.mod(x - lo, 2*W)
            x = np.where(finite, np.where(y <= W, y, 2*W - y) + lo, x)
        return self.start + x[:, :, None]*self.d0

    def clear(self, pts, ts, margin):
        if self.room is not None:
            keep = ROBOT_RADIUS + self.wall_margin
            if np.any(pts < self.room[:2] + keep) or np.any(pts > self.room[2:] - keep):
                return False
        if len(self.radius) == 0:
            return True
        gaps = np.linalg.norm(pts[:, None] - self.predict(ts), axis=2)
        need = ROBOT_RADIUS + margin + self.radius - self.slack[None]*np.clip(1 - ts[:, None]/2.0, 0, 1)
        return bool(np.all(gaps >= need))

    def connect(self, x, y, yaw, t, goal, margin):
        """Spin to face the goal, drive straight to it: rows (t, x, y, yaw, v, w), or None if it would hit something."""
        gap = goal - [x, y]
        dist = np.linalg.norm(gap)
        turn = wrap(np.arctan2(gap[1], gap[0]) - yaw)
        n_spin, n_run = int(np.ceil(abs(turn)/self.w_plan/self.sample_dt)), int(np.ceil(dist/self.v_plan/self.sample_dt))
        ts = t + self.sample_dt*np.arange(1, n_spin + n_run + 1)
        frac_spin = np.arange(1, n_spin + 1)/max(n_spin, 1)
        frac_run = np.arange(1, n_run + 1)/max(n_run, 1)
        xs = np.concatenate([np.full(n_spin, x), x + gap[0]*frac_run])
        ys = np.concatenate([np.full(n_spin, y), y + gap[1]*frac_run])
        yaws = np.concatenate([yaw + turn*frac_spin, np.full(n_run, yaw + turn)])
        if not self.clear(np.stack([xs, ys], axis=1), ts, margin):
            return None
        vs = np.concatenate([np.zeros(n_spin), np.full(n_run, self.v_plan)])
        ws = np.concatenate([np.full(n_spin, np.sign(turn)*self.w_plan), np.zeros(n_run)])
        return np.stack([ts, xs, ys, yaws, vs, ws], axis=1)

    def search(self, p, yaw, goal, margin, rng):
        """RRT from (p, yaw) at time 0. Returns (rows, reached_goal), or (None, False) if not even one edge is free."""
        here = self.predict(np.zeros(1))[0] if len(self.radius) else np.zeros((0, 2))
        self.slack = np.maximum(ROBOT_RADIUS + margin + self.radius + 0.02 - np.linalg.norm(here - p, axis=1), 0.0)
        self.wall_margin = margin
        if self.room is not None:   # never ask for more wall clearance than the start or the goal has
            room_gap = np.minimum(np.minimum(p, goal) - self.room[:2], self.room[2:] - np.maximum(p, goal)).min()
            self.wall_margin = float(np.clip(room_gap - ROBOT_RADIUS - 0.01, 0.0, margin))
        direct = self.connect(*p, yaw, 0.0, goal, margin)
        if direct is not None and not (self.speed > 0).any():
            return direct, True
        straight_time = np.linalg.norm(goal - p)/self.v_plan
        horizon = straight_time + self.horizon_extra
        nodes, parents, edges = [np.array([*p, yaw, 0.0])], [-1], [np.zeros((0, 6))]
        best, best_tail = None, direct
        best_t = direct[-1, 0] if direct is not None else np.inf
        lo, hi = np.minimum(p, goal) - 2.0, np.maximum(p, goal) + 2.0
        if self.room is not None:
            lo, hi = np.maximum(lo, self.room[:2]), np.minimum(hi, self.room[2:])

        def add(i, rows):
            nonlocal best, best_tail, best_t
            nodes.append(np.array([rows[-1, 1], rows[-1, 2], rows[-1, 3], rows[-1, 0]]))
            parents.append(i)
            edges.append(rows)
            tail = self.connect(*nodes[-1], goal, margin)
            if tail is not None and tail[-1, 0] < best_t:
                best, best_tail, best_t = len(nodes) - 1, tail, tail[-1, 0]

        def wait_from(i, d):
            node = nodes[i]
            ts = node[3] + self.sample_dt*np.arange(1, int(round(d/self.sample_dt)) + 1)
            pts = np.tile(node[:2], (len(ts), 1))
            if ts[-1] <= horizon and self.clear(pts, ts, margin):
                add(i, np.column_stack([ts, pts, np.full(len(ts), node[2]), np.zeros((len(ts), 2))]))
                return True
            return False

        movers = bool((self.speed > 0).any())
        if movers:
            for d in self.wait_choices:   # stage 1: waiting at the start, then going straight
                wait_from(0, d)
        found_at = None if not np.isfinite(best_t) else 0
        for it in range(self.iters):
            if found_at is None and np.isfinite(best_t):
                found_at = it
            if found_at is not None and it - found_at >= self.iters_after_found:
                break
            A = np.array(nodes)
            if movers and rng.random() < 0.15:
                i = int(rng.integers(len(nodes)))
                wait_from(i, self.wait_choices[int(rng.integers(len(self.wait_choices)))])
                continue
            s_xy = goal if rng.random() < 0.25 else rng.uniform(lo, hi)
            s_t = rng.uniform(0, horizon)
            i = int(np.argmin(np.linalg.norm(A[:, :2] - s_xy, axis=1) + 0.05*np.abs(A[:, 3] - s_t)))
            x, y, th, t = nodes[i]
            c, si = np.cos(th), np.sin(th)
            ex = x + c*self.edges[:, :, 0] - si*self.edges[:, :, 1]
            ey = y + si*self.edges[:, :, 0] + c*self.edges[:, :, 1]
            ts = t + self.sample_dt*np.arange(1, self.edge_steps + 1)
            if ts[-1] > horizon:
                continue
            order = np.argsort(np.hypot(ex[:, -1] - s_xy[0], ey[:, -1] - s_xy[1]))
            for k in order[:3]:
                pts = np.stack([ex[k], ey[k]], axis=1)
                if self.clear(pts, ts, margin):
                    add(i, np.column_stack([ts, pts, th + self.edges[k, :, 2],
                                            np.full((self.edge_steps, 2), self.controls[k])]))
                    break
        def chain(i):
            parts = []
            while i > 0:
                parts.append(edges[i]); i = parents[i]
            return parts[::-1]
        if np.isfinite(best_t):
            parts = chain(best) if best is not None else []
            return self.shorten(np.vstack(parts + [best_tail]), p, yaw, goal, margin, rng), True
        if len(nodes) == 1:
            return None, False
        # Goal not reached: keep the plan that stays safe longest, ending as close to the goal as possible
        i = min(range(1, len(nodes)), key=lambda k: (-round(nodes[k][3], 1), np.linalg.norm(nodes[k][:2] - goal)))
        return np.vstack(chain(i)), False

    def rebuild(self, p, yaw, stops, margin):
        """Rows for "wait, then turn and drive straight" through stops [(x, y), wait_s], or None if anything is hit."""
        x, y, th, t, parts = p[0], p[1], yaw, 0.0, []
        for (q, wait) in stops:
            if wait > 0:
                ts = t + self.sample_dt*np.arange(1, int(round(wait/self.sample_dt)) + 1)
                pts = np.tile([x, y], (len(ts), 1))
                if not self.clear(pts, ts, margin):
                    return None
                parts.append(np.column_stack([ts, pts, np.full(len(ts), th), np.zeros((len(ts), 2))]))
                t = ts[-1]
            if np.hypot(q[0] - x, q[1] - y) > 1e-3:
                rows = self.connect(x, y, th, t, np.asarray(q), margin)
                if rows is None:
                    return None
                parts.append(rows)
                t, x, y, th = rows[-1, :4]
        return np.vstack(parts) if parts else None

    def shorten(self, rows, p, yaw, goal, margin, rng):
        """Standard RRT clean-up. Reduce the plan to stops every edge, then repeatedly try to skip stops or trim a
        wait, keeping a change only if the plan is still collision-free at its new times and arrives earlier."""
        stops, wait = [], 0.0
        for k, row in enumerate(rows):
            if row[4] == 0 and row[5] == 0:
                wait += self.sample_dt
            elif (k + 1) % self.edge_steps == 0:
                stops.append([row[1:3].copy(), round(wait, 1)])
                wait = 0.0
        stops.append([goal.copy(), round(wait, 1)])
        best, best_t = rows, rows[-1, 0]
        for _ in range(self.shorten_iters):
            trial = [list(q) for q in stops]
            waits = [k for k, q in enumerate(trial) if q[1] > 0]
            if waits and rng.random() < 0.5:
                k = waits[int(rng.integers(len(waits)))]
                trial[k][1] = round(max(trial[k][1] - self.wait_trim, 0.0), 1)
            elif len(trial) > 1:
                i = int(rng.integers(len(trial) - 1))
                j = int(rng.integers(i + 1, len(trial)))
                # Skip stops i..j-1, but keep any waiting they contained until the mover check says it can go
                trial = trial[:i] + [[trial[j][0], sum(q[1] for q in trial[i:j + 1])]] + trial[j + 1:]
            else:
                break
            built = self.rebuild(p, yaw, trial, margin)
            if built is not None and built[-1, 0] <= best_t + 1e-9:
                stops, best, best_t = trial, built, built[-1, 0]
        return best

    def make_plan(self, p, yaw, goal):
        self.load_world()
        self.plans += 1
        rng = np.random.default_rng(int(self.now*1000))
        fallback = None
        for margin in self.margins:
            rows, reached = self.search(p, yaw, goal, margin, rng)
            if reached:
                return self.store(rows, p, yaw, goal, margin)
            fallback = fallback or (rows, margin) if rows is not None else fallback
        self.plan_failures += 1
        return self.store(fallback[0], p, yaw, goal, fallback[1]) if fallback else None

    def store(self, rows, p, yaw, goal, margin):
        rows = np.vstack([[0.0, *p, yaw, 0.0, 0.0], rows]).copy()
        rows[:, 0] += self.now
        rows[:, 3] = np.unwrap(rows[:, 3])
        return {"rows": rows, "goal": goal.copy(), "margin": margin}

    def ref(self, t):
        rows = self.plan["rows"]
        i = min(int(np.searchsorted(rows[:, 0], t)), len(rows) - 1)
        return np.array([np.interp(t, rows[:, 0], rows[:, k]) for k in (1, 2, 3)]), rows[i, 4], rows[i, 5]

    def plan_ok(self, p, goal):
        rows = self.plan["rows"]
        if self.now >= rows[-1, 0] or np.linalg.norm(rows[-1, 1:3] - goal) > 0.3:
            return False
        return np.linalg.norm(self.ref(self.now)[0][:2] - p) <= self.max_stray

    def wheels(self, p, yaw, velocity, lidar, target):
        to_target = target[:2] - p
        goal = target[:2] - standoff(target[2]) * to_target / max(np.linalg.norm(to_target), 1e-9)
        dist = np.linalg.norm(goal - p)
        self.parking = dist < (self.park_exit if self.parking else self.park_handover)
        self.waiting = False
        if self.parking:     # no lidar: the plan brought it here, and the generator keeps movers off the spot
            return super().wheels(p, yaw, velocity, np.full(LIDAR_RAYS, LIDAR_RANGE), target)
        if self.now >= self.next_check:
            self.next_check = self.now + self.check_every
            if self.plan is None or not self.plan_ok(p, goal):
                self.plan = self.make_plan(p, yaw, goal)
        if self.plan is None:   # nowhere safe to go right now
            self.waiting = True
            return np.zeros(4), False
        pose, v_ref, w_ref = self.ref(self.now + 2*self.sample_dt)
        err = pose[:2] - p
        forward = np.cos(yaw)*err[0] + np.sin(yaw)*err[1]
        left = -np.sin(yaw)*err[0] + np.cos(yaw)*err[1]
        turn = wrap(pose[2] - yaw)
        if v_ref == 0 and w_ref == 0 and np.hypot(forward, left) < 0.1 and abs(turn) < 0.1:
            self.waiting = True
            return np.zeros(4), False
        v = float(np.clip(v_ref + 1.0*forward, -self.max_v, self.max_v))
        w = float(np.clip(w_ref + 2.0*turn + 2.0*left*np.sign(v_ref or 1.0), -self.max_spin, self.max_spin))
        l, r = (v - w*self.track/2)/self.wheel_radius, (v + w*self.track/2)/self.wheel_radius
        return np.array([r, l, l, r]), False

class ObstacleRobot(Robot):
    def __init__(self, config, scenario, gui=False):
        super().__init__(config, scenario, gui)
        p, client = self.p, self.client
        # Course physics give the wheels 1% sideways grip; then a sustained gentle curve (needed to drive around
        # obstacles) builds a sideways slide that grows past 1 m/s. At 10% it stays below about 0.1 m/s.
        # Obstacle runs only: robot_env.py and the course base task keep 0.01.
        for j in range(client.getNumJoints(self.rid)):
            if "wheel" in client.getJointInfo(self.rid, j)[1].decode():
                client.changeDynamics(self.rid, j, anisotropicFriction=[1.0, WHEEL_SIDE_GRIP, 1.0])
        self.obstacle_ids = []
        # Optional per-obstacle "vel" [vx, vy] (m/s) and "span" (m travelled along vel before turning round).
        # Obstacles without "vel" stay where they are, so every existing scenario file behaves as before.
        self.obstacle_xy = np.array([o["xy"] for o in scenario["obstacles"]], dtype=float).reshape(-1, 2)
        self.obstacle_vel = np.array([o.get("vel", [0.0, 0.0]) for o in scenario["obstacles"]], dtype=float).reshape(-1, 2)
        # Signed distance from the start along the first direction at which a mover turns round
        self.obstacle_sweep = np.array([o.get("sweep", [-o.get("span", np.inf), o.get("span", np.inf)])
                                        for o in scenario["obstacles"]], dtype=float).reshape(-1, 2)
        self.obstacle_start = self.obstacle_xy.copy()
        self.obstacle_r = np.array([o["r"] for o in scenario["obstacles"]], dtype=float)
        self.obstacle_vel0 = self.obstacle_vel.copy()   # the reversal below flips obstacle_vel
        self.has_moving_obstacles = bool(np.any(self.obstacle_vel != 0))
        for o in scenario["obstacles"]:
            col = client.createCollisionShape(p.GEOM_CYLINDER, radius=o["r"], height=OBSTACLE_HEIGHT)
            vis = client.createVisualShape(p.GEOM_CYLINDER, radius=o["r"], length=OBSTACLE_HEIGHT,
                                           rgbaColor=[0.45, 0.45, 0.5, 1])
            self.obstacle_ids.append(client.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis, basePosition=[o["xy"][0], o["xy"][1], OBSTACLE_HEIGHT/2]))
        self.room, self.wall_ids = None, []
        if "room" in scenario:
            self.room = np.array(scenario["room"], dtype=float)
            (x0, y0, x1, y1), t = self.room, WALL_THICKNESS
            for cx, cy, hx, hy in [((x0 + x1)/2, y0 - t/2, (x1 - x0)/2 + t, t/2), ((x0 + x1)/2, y1 + t/2, (x1 - x0)/2 + t, t/2),
                                   (x0 - t/2, (y0 + y1)/2, t/2, (y1 - y0)/2 + t), (x1 + t/2, (y0 + y1)/2, t/2, (y1 - y0)/2 + t)]:
                # The wall itself is invisible (it still stops the robot and the lidar); only a flat line shows
                col = client.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, WALL_HEIGHT/2])
                self.wall_ids.append(client.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=-1, basePosition=[cx, cy, WALL_HEIGHT/2]))
                vis = client.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, 0.005], rgbaColor=[0.9, 0.2, 0.1, 1])
                client.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis,
                                       basePosition=[cx, cy, 0.005])
            if gui:   # look at the whole room
                client.resetDebugVisualizerCamera(float(max(x1 - x0, y1 - y0)), 30, -60, [(x0 + x1)/2, (y0 + y1)/2, 0])
        x, y, yaw = scenario["start"]
        z = client.getBasePositionAndOrientation(self.rid)[0][2]
        client.resetBasePositionAndOrientation(self.rid, [x, y, z], p.getQuaternionFromEuler([0, 0, yaw]))
        client.resetBaseVelocity(self.rid, [0, 0, 0], [0, 0, 0])
        if "arm_q0" in scenario:   # optional arm start pose (7 angles); the expert then holds this pose while driving
            arm_joints = {client.getJointInfo(self.rid, j)[1].decode(): j for j in range(client.getNumJoints(self.rid))}
            for k, angle in enumerate(scenario["arm_q0"]):
                client.resetJointState(self.rid, arm_joints[f"robot_j2s7s300_joint_{k+1}"], angle, 0.0)
        self.sim.ReceiveObservation()
        if "arm_q0" in scenario:
            self.q_init = np.array(self.sim.GetMotorAngles(0))[4:11].copy()
            self.q_des = self.q_init.copy()
        # sensor-only either way: never sees scenario["obstacles"]
        self.base_expert = RRTExpert(self) if scenario.get("expert") == "rrt" else SteerAround()
        self.base_params.nav_timeout = NAV_TIMEOUT
        self.arrived = False

    def advance_obstacles(self, dt):
        """Move the moving obstacles by dt along their velocity; after `span` metres they turn round.
        SteerAround reads the lidar, never these positions; RRTExpert reads them."""
        if not self.has_moving_obstacles:
            return
        self.obstacle_xy += self.obstacle_vel*dt
        speed0 = np.linalg.norm(self.obstacle_vel0, axis=1)
        d0 = self.obstacle_vel0 / np.maximum(speed0, 1e-9)[:, None]
        along = np.sum((self.obstacle_xy - self.obstacle_start)*d0, axis=1)
        heading = np.sum(self.obstacle_vel*d0, axis=1)
        turn = ((along > self.obstacle_sweep[:, 1]) & (heading > 0)) | ((along < self.obstacle_sweep[:, 0]) & (heading < 0))
        self.obstacle_vel[turn] *= -1
        for k, body in enumerate(self.obstacle_ids):
            if np.any(self.obstacle_vel[k] != 0):
                self.client.resetBasePositionAndOrientation(
                    body, [self.obstacle_xy[k, 0], self.obstacle_xy[k, 1], OBSTACLE_HEIGHT/2], [0, 0, 0, 1])

    def lidar(self, state):
        """36 distances (m, from the base centre) to the nearest obstacle in each direction; LIDAR_RANGE if none."""
        base, yaw = state["base"], state["yaw"]
        u = np.stack([np.cos(yaw + LIDAR_ANGLES), np.sin(yaw + LIDAR_ANGLES), np.zeros(LIDAR_RAYS)], axis=1)
        origin = np.array([base[0], base[1], base[2] + LIDAR_HEIGHT])
        hits = self.client.rayTestBatch((origin + LIDAR_START*u).tolist(), (origin + LIDAR_RANGE*u).tolist())
        return np.array([LIDAR_START + h[2]*(LIDAR_RANGE - LIDAR_START) if h[0] >= 0 else LIDAR_RANGE
                         for h in hits])

    def measure(self):
        state = super().measure()
        state["lidar"] = self.lidar(state)
        return state

    def contacts(self):
        """Referee only: for each obstacle, whether the robot is touching it right now."""
        return np.array([bool(self.client.getContactPoints(self.rid, o)) for o in self.obstacle_ids], dtype=bool)

    def touching(self):
        return any(self.client.getContactPoints(self.rid, o) for o in self.obstacle_ids + self.wall_ids)

    def touching_wall(self):
        return any(self.client.getContactPoints(self.rid, w) for w in self.wall_ids)

    def clearance(self, up_to=2.0):
        """Referee only: gap (m) between the robot's body (any link) and the nearest obstacle surface."""
        gaps = [c[8] for o in self.obstacle_ids + self.wall_ids for c in self.client.getClosestPoints(self.rid, o, up_to)]
        return float(min(gaps, default=up_to))

    def expert_action(self, state, elapsed):
        if hasattr(self.base_expert, "observe"):
            self.base_expert.observe(elapsed, state["base"][:2], state["yaw"], state["lidar"])
        if not self.arrived:
            wheel, self.arrived = self.base_expert.wheels(state["base"][:2], state["yaw"], state["x"][22:24],
                                                          state["lidar"], self.target)
            if elapsed > self.base_params.nav_timeout:
                self.base_timed_out = True
            if self.arrived or self.base_timed_out:
                wheel = np.zeros(4)
        else:
            wheel = np.zeros(4)
        # After a navigation timeout the base stops and the arm keeps holding: reaching for a target that may
        # be metres away only swings the arm into obstacles. The episode then ends as a timeout.
        torque, self.q_des, _ = self.expert.arm_controller(self.dyn, state["q_pin"],
            state["q"][4:11], state["qd"][4:11], self.q_des, self.q_init, self.target,
            state["ee"], self.ee_id, self.shoulder_id, self.lower, self.upper, self.control_dt,
            self.arm_params, reach_active=self.arrived)
        return np.concatenate([wheel, torque])

def rollout(config, scenario, folder, policy=None, gui=False, label_expert=False, lost_distance=None, save_every=1,
            perturb=None, label_arm=False):
    """Like robot_env.rollout, plus lidar, collisions and base arrival. policy.act(state) -> 11 commands.

    label_expert (DAgger): while a learned policy drives, also ask the base expert what wheel speeds it would use
    in each visited state and save them as "expert_wheels". They never move the robot. This works because the
    expert is memoryless and sensor-only: it can label any state from the current sensors alone.
    label_arm (arm DAgger): also save "expert_arm", the 7 torques a SHADOW copy of the course arm expert would
    command. The arm expert has memory (its reference pose q_des), so the shadow keeps its own q_des: it starts at
    the start pose like the expert's, starts reaching when the policy does (policy.reach_from is set; for the
    expert itself, when it arrives) and is updated every step from the real arm's measured state. The expert caps
    q_des at 0.15 rad from the measured angles, so the shadow stays anchored to the real arm. Never moves the robot.
    A policy's reach_from is also saved as base_arrival_time_s.
    lost_distance: end the episode early ("lost") once the base is this far (m) from the target.
    save_every: keep every Nth control step in trajectory.npz (the summary still uses every step).
    perturb (robustness tests), all optional: lidar_noise (m, Gaussian per ray), lidar_dropout (chance a ray
    reports "nothing"), push_n (N, sideways shove to the robot's left) with push_at_s / push_dur_s,
    delay_ms (commands reach the motors this late, zero until the first one arrives; "applied_actions" records
    what the motors actually got; delay_part "wheels" (default) or "all": the stiff arm PD
    becomes unstable with ~20 ms delay, which would hide the base result), push_sign (+1 left, -1 right), seed. Noise only changes what the controller SEES
    (policy, or the expert when policy is None); recorded observations, the referee and DAgger labels
    (label_expert) use the true sensors. When lidar is perturbed the seen lidar is saved as "lidar_seen"."""
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(f"Refusing to overwrite episode: {folder}")
    folder.mkdir(parents=True)
    robot = None
    try:
        with (folder/"simulator.log").open("w") as log, contextlib.redirect_stdout(log):
            robot = ObstacleRobot(config, scenario, gui)
            keys = ["observations", "lidar", "actions", "applied_actions", "raw_actions", "time_s", "ee_m",
                    "error_m", "base_m", "inference_ms", "arm_q_rad", "contact"] + (["expert_wheels"] if label_expert else []) \
                   + (["expert_arm"] if label_arm else []) + (["obstacle_xy"] if robot.has_moving_obstacles else []) \
                   + (["contact_obstacle"] if robot.has_moving_obstacles else []) \
                   + (["waiting"] if hasattr(robot.base_expert, "waiting") else []) \
                   + (["contact_wall"] if robot.wall_ids else [])
            shadow_q_des = robot.q_init.copy()
            pert = perturb or {}
            noisy_lidar = pert.get("lidar_noise", 0) > 0 or pert.get("lidar_dropout", 0) > 0
            if noisy_lidar:
                keys.append("lidar_seen")
            rows = {k: [] for k in keys}
            rng = np.random.default_rng(pert.get("seed", 0))
            delay_steps = int(round(pert.get("delay_ms", 0) / (1000*robot.control_dt)))
            pending = collections.deque()
            elapsed, dwell, clipped, reason = 0.0, 0.0, 0, "timeout"
            arrival_time = None
            min_clearance = robot.clearance()
            state = robot.measure()
            while elapsed < config["duration_s"]:
                start = time.perf_counter()
                seen = state
                if noisy_lidar:
                    lid = state["lidar"] + rng.normal(0, pert.get("lidar_noise", 0), LIDAR_RAYS)
                    lid[rng.random(LIDAR_RAYS) < pert.get("lidar_dropout", 0)] = LIDAR_RANGE
                    seen = {**state, "lidar": np.clip(lid, LIDAR_START, LIDAR_RANGE)}
                raw = robot.expert_action(seen, elapsed) if policy is None else policy.act(seen)
                latency = (time.perf_counter() - start)*1000
                action = clip_action(raw, config)
                clipped += int(np.any(np.abs(raw - action) > 1e-10))
                hit = robot.contacts()
                for k, v in {"observations": state["x"], "lidar": state["lidar"], "actions": action,
                             "raw_actions": raw, "time_s": elapsed, "ee_m": state["ee"], "error_m": state["error"],
                             "base_m": state["base"], "inference_ms": latency, "arm_q_rad": state["q"][4:11],
                             "contact": hit.any() or robot.touching_wall()}.items():
                    rows[k].append(np.copy(v))
                if noisy_lidar:
                    rows["lidar_seen"].append(np.copy(seen["lidar"]))
                if robot.has_moving_obstacles:   # ground truth for scoring a tracker; never given to a controller
                    rows["obstacle_xy"].append(robot.obstacle_xy.copy())
                    rows["contact_obstacle"].append(hit.copy())
                if "contact_wall" in rows:
                    rows["contact_wall"].append(robot.touching_wall())
                if "waiting" in rows:
                    rows["waiting"].append(robot.base_expert.waiting)
                if label_expert:
                    label, parked = robot.base_expert.wheels(state["base"][:2], state["yaw"], state["x"][22:24],
                                                             state["lidar"], robot.target)
                    rows["expert_wheels"].append(np.zeros(4) if parked else label)
                if label_arm:
                    reaching = robot.arrived if policy is None else getattr(policy, "reach_from", None) is not None
                    torque, shadow_q_des, _ = robot.expert.arm_controller(robot.dyn, state["q_pin"],
                        state["q"][4:11], state["qd"][4:11], shadow_q_des, robot.q_init, robot.target, state["ee"],
                        robot.ee_id, robot.shoulder_id, robot.lower, robot.upper, robot.control_dt,
                        robot.arm_params, reach_active=reaching)
                    rows["expert_arm"].append(clip_action(np.concatenate([np.zeros(4), torque]), config)[4:])   # same limits as the expert data
                pending.append(action)
                # Delay: until the first command arrives the motors get nothing (the robot starts at rest)
                applied = pending.popleft() if len(pending) > delay_steps else np.zeros_like(action)
                if pert.get("delay_part", "wheels") == "wheels":   # arm torques stay on time
                    applied = np.concatenate([applied[:4], action[4:]])
                rows["applied_actions"].append(np.copy(applied))
                push_at = pert.get("push_at_s", 6.0)
                if pert.get("push_n", 0) and push_at <= elapsed < push_at + pert.get("push_dur_s", 0.3):
                    left = pert.get("push_sign", 1) * np.array([-np.sin(state["yaw"]), np.cos(state["yaw"]), 0.0])
                    robot.client.applyExternalForce(robot.rid, -1, (pert["push_n"]*left).tolist(),
                                                    state["base"].tolist(), robot.p.WORLD_FRAME)
                robot.step(applied)
                robot.advance_obstacles(robot.control_dt)
                if len(rows["time_s"]) % 10 == 0:     # every 10 ms: closest-point queries are slow
                    min_clearance = min(min_clearance, robot.clearance())
                elapsed += robot.control_dt
                state = robot.measure()
                if policy is None and robot.arrived and arrival_time is None:
                    arrival_time = elapsed
                if policy is not None and arrival_time is None and getattr(policy, "reach_from", None) is not None:
                    arrival_time = policy.reach_from
                dwell = dwell + robot.control_dt if state["error"] <= config["success_tolerance_m"] else 0.0
                if elapsed >= config["min_success_time_s"] and dwell >= config["success_hold_s"]:
                    reason = "success"
                    break
                if lost_distance and np.linalg.norm(robot.target[:2] - state["base"][:2]) > lost_distance:
                    reason = "lost"
                    break
            arrays = {k: np.asarray(v) for k, v in rows.items()}
            path = np.vstack([arrays["base_m"], state["base"]])
            to_target = robot.target[:2] - state["base"][:2]
            standoff_error = float(abs(np.linalg.norm(to_target) - standoff(robot.target[2])))
            facing_error = float(abs(wrap(np.arctan2(to_target[1], to_target[0]) - state["yaw"])))
            # contact is sampled before each step, so also check the state the last step left behind
            contact_steps = int(arrays["contact"].sum()) + int(robot.touching())
            summary = {"scenario_id": scenario["id"], "split": scenario["split"],
                "controller": "expert" if policy is None else "learned",
                "success": reason == "success" and contact_steps == 0, "reached_target": reason == "success",
                "termination": reason, "simulation_time_s": elapsed, "final_error_m": state["error"],
                "minimum_error_m": float(min(np.min(arrays["error_m"]), state["error"])),
                "final_standoff_error_m": standoff_error, "final_facing_error_rad": facing_error,
                "base_arrival_time_s": arrival_time,
                "contact_steps": contact_steps,
                **({"wall_contact_steps": int(arrays["contact_wall"].sum())} if robot.wall_ids else {}),
                **({"contact_steps_per_obstacle": arrays["contact_obstacle"].sum(axis=0).tolist(),
                    "obstacle_is_moving": [bool(np.any(v != 0)) for v in robot.obstacle_vel0]}
                   if robot.has_moving_obstacles else {}), "contact_fraction": contact_steps/len(arrays["time_s"]),
                "min_lidar_m": float(arrays["lidar"].min()), "min_clearance_m": min_clearance,
                "base_path_length_m": float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum()),
                "clipped_action_fraction": clipped/len(arrays["time_s"]),
                "base_nav_timeout_triggered": bool(policy is None and robot.base_timed_out),
                "control_period_s": robot.control_dt, "samples": len(arrays["time_s"]), "perturb": pert}
            np.savez_compressed(folder/"trajectory.npz", **{k: v[::save_every] for k, v in arrays.items()})
            if isinstance(robot.base_expert, RRTExpert):
                summary["wait_time_s"] = robot.base_expert.wait_time
                summary["plans"], summary["plan_failures"] = robot.base_expert.plans, robot.base_expert.plan_failures
            (folder/"summary.json").write_text(json.dumps(summary, indent=2))
            (folder/"episode.json").write_text(json.dumps({"scenario": scenario, "settings": config,
                "source_sha256": provenance(config), "final_ee_m": state["ee"].tolist(),
                "lidar": {"rays": LIDAR_RAYS, "start_m": LIDAR_START, "range_m": LIDAR_RANGE, "height_m": LIDAR_HEIGHT},
                "obstacle_height_m": OBSTACLE_HEIGHT, "wheel_side_grip": WHEEL_SIDE_GRIP,
                "saved_every": save_every}, indent=2))
        return summary
    except Exception as error:
        (folder/"failure.json").write_text(json.dumps({"error": repr(error), "scenario": scenario}, indent=2))
        raise
    finally:
        if robot is not None:
            robot.close()
