"""Obstacle world: static cylinders, a random start pose, a 2D lidar and a steer-around expert for the base.

Builds on robot_env.Robot (unchanged) and the course arm expert. The base expert is ours:
a memoryless "steer-around" (vortex) field, so its wheel command depends only on what it senses right now,
never on an internal mode. It is sensor-only: lidar, own pose/velocity and the target position, exactly what
a learned policy gets. Obstacle positions are used only to build the world and to count contacts (referee).
Exception: RRTExpert deliberately reads the true obstacles (teacher-student learning); VisibleRRTExpert only those the lidar hit.
Physics change for obstacle runs only: wheel sideways grip 0.1 instead of the course's 0.01 (see ObstacleRobot).
"""
import collections
import contextlib
import heapq
import itertools
import json
import time
from pathlib import Path
import numpy as np
from scipy import ndimage
from models import clip_action
from reach_zone import parking_distance
from robot_env import Robot, provenance

ROBOT_RADIUS = 0.47        # m: circle around the base footprint (wheel boxes reach x 0.342, y 0.313)
OBSTACLE_HEIGHT = 0.6      # m: taller than the lidar plane
LIDAR_RAYS = 36            # default: one ray every 10 degrees, ray 0 = straight ahead, counter-clockwise
                           # (config "lidar_rays" changes it per run, e.g. 72 = every 5 degrees; see lidar_angles)
LIDAR_START = 0.5          # m from base centre: rays start just outside the robot body
LIDAR_RANGE = 4.0          # m from base centre
LIDAR_HEIGHT = 0.3         # m above the base frame
LIDAR_ANGLES = np.arange(LIDAR_RAYS) * 2*np.pi/LIDAR_RAYS


def lidar_angles(rays):
    """Ray directions (rad, robot frame) for a lidar with this many evenly spaced rays; LIDAR_ANGLES for 36."""
    return np.arange(rays) * 2*np.pi/rays
WHEEL_SIDE_GRIP = 0.1      # sideways wheel friction factor; the course uses 0.01 (see ObstacleRobot)
NAV_TIMEOUT = 50.0         # s: detours with stop-turn-go take longer than the course's 25 s open-ground limit
REACH_BAND = 0.2   # m: within this of the reach distance, the base may stop and creep instead of going to its chosen spot
MOVER_CROSS_COST = 2.0   # m added to a leg whose route crosses a mover's sweep
ROOM_CHANGE_COST, ORDER_TIE = 2.5, 0.1     # m added per room change when ordering dots; share of the best cost counted as a tie
RETRACT_RATE, RETRACT_SETTLED = 0.5, 0.05   # rad/s: speed of the arm's return between dots; joint speed counted as stopped
ARM_START = [0.0, 1.57, 0.0, 1.0, 0.0, 1.0, 0.0]   # the arm's usual start pose (no scenario arm_q0)
RRT_NAV_TIMEOUT = 90.0     # s: RRTExpert also waits for movers, so its routes can take longer
STILL_SPEED = 0.02         # m/s: slower than this counts as not moving for stall_limit_s

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

def box_gaps(points, lo, hi):
    """Distance from points (..., 2) to axis-aligned boxes lo, hi (boxes, 2): array (..., boxes), 0 inside a box."""
    d = np.maximum(np.maximum(lo - points[..., None, :], points[..., None, :] - hi), 0.0)
    return np.linalg.norm(d, axis=-1)

class FloorPlanRoute:
    """A* on a grid over the room, around walls, furniture and static cylinders, to guide RRTExpert between rooms.
    RRT alone rarely finds a doorway; here it only has to reach the farthest route point it can see. Movers are
    left to RRT. A route with room to turn (`wide`, which still fits a 1.4 m doorway) is preferred over one through a
    tight gap, where the arm cannot swing round, unless the tight route saves more than `tight_cost` metres."""
    cell = 0.1            # m
    wide, narrow = ROBOT_RADIUS + 0.2, ROBOT_RADIUS + 0.08   # m kept from anything
    tight_cost = 1.5      # m a tight route must save to be chosen
    blocked_start = 0.35  # m of a line from an already tight spot that may be in the kept-clear zone

    def __init__(self, room, lo, hi, centres, radii):
        self.origin = np.asarray(room[:2], dtype=float)
        n = np.ceil((np.asarray(room[2:]) - self.origin)/self.cell).astype(int) + 1
        xs = self.origin[0] + self.cell*np.arange(n[0])
        ys = self.origin[1] + self.cell*np.arange(n[1])
        grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)
        gap = np.minimum(grid - self.origin, np.asarray(room[2:]) - grid).min(axis=-1)
        if len(lo):
            gap = np.minimum(gap, box_gaps(grid, lo, hi).min(axis=-1))
        for c, r in zip(centres, radii):
            gap = np.minimum(gap, np.linalg.norm(grid - c, axis=-1) - r)
        self.gap = gap
        self.free = gap >= self.wide
        self.field_key, self.field = None, None
        self.grid = grid

    def set_rooms(self, wall_lo, wall_hi):
        """Number the rooms: the areas left when the walls are thickened enough to close the 1.4 m doorways."""
        open_area = box_gaps(self.grid, wall_lo, wall_hi).min(axis=-1) > 0.75 if len(wall_lo) else np.ones(self.gap.shape, dtype=bool)
        self.rooms, _ = ndimage.label(open_area)

    def room_of(self, p):
        cell = np.array(self.to_cell(p))
        idx = np.argwhere(self.rooms > 0)
        if len(idx) == 0:
            return 0
        return int(self.rooms[tuple(idx[np.argmin(np.sum((idx - cell)**2, axis=1))])])

    def to_cell(self, p):
        return tuple(np.clip(np.round((np.asarray(p) - self.origin)/self.cell).astype(int), 0, np.array(self.free.shape) - 1))

    def nearest_free(self, cell):
        if self.free[cell]:
            return cell
        idx = np.argwhere(self.free)
        return tuple(idx[np.argmin(np.sum((idx - np.array(cell))**2, axis=1))]) if len(idx) else None

    def sees(self, a, b):
        """Is the straight line a-b free on the grid?"""
        length = float(np.linalg.norm(np.asarray(b) - a))
        n = max(2, int(length/(self.cell/2)) + 1)
        for u in np.linspace(0, 1, n):
            if not self.free[self.to_cell(np.asarray(a) + u*(np.asarray(b) - a))] and u*length > self.blocked_start:
                return False
        return True

    def path(self, start, goal):
        """Route points from start to goal (world coordinates), or None if there is no route."""
        length = lambda pts: float(np.sum(np.linalg.norm(np.diff(np.vstack([start, *pts]), axis=0), axis=1)))
        self.free = self.gap >= self.narrow
        tight = self.search(start, goal)
        self.free = self.gap >= self.wide
        pts = self.search(start, goal)
        if tight is not None and (pts is None or length(tight) + self.tight_cost < length(pts)):
            self.free = self.gap >= self.narrow
            return tight
        return pts

    def lengths_from(self, start):
        """Shortest route length (m) from start to every grid cell, with the same preference as path()."""
        key = tuple(np.round(start, 2))
        if self.field_key != key:
            self.field_key = key
            self.field = np.minimum(self.flood(start, self.gap >= self.wide), self.flood(start, self.gap >= self.narrow) + self.tight_cost)
        return self.field

    def flood(self, start, free):
        self.free = free
        a = self.nearest_free(self.to_cell(start))
        dist = np.full(free.shape, np.inf)
        if a is None:
            return dist
        dist[a] = 0.0
        heap = [(0.0, a)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                v = (u[0] + dx, u[1] + dy)
                if 0 <= v[0] < free.shape[0] and 0 <= v[1] < free.shape[1] and free[v]:
                    c = d + self.cell*np.hypot(dx, dy)
                    if c < dist[v]:
                        dist[v] = c
                        heapq.heappush(heap, (c, v))
        return dist

    def search(self, start, goal):
        a, b = self.nearest_free(self.to_cell(start)), self.nearest_free(self.to_cell(goal))
        if a is None or b is None:
            return None
        best, parent, heap = {a: 0.0}, {}, [(0.0, a)]
        while heap:
            _, u = heapq.heappop(heap)
            if u == b:
                cells = [u]
                while cells[-1] in parent:
                    cells.append(parent[cells[-1]])
                pts = [self.origin + self.cell*np.array(c) for c in cells[::-1]]
                return pts[:-1] + [np.asarray(goal, dtype=float)]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                v = (u[0] + dx, u[1] + dy)
                if not (0 <= v[0] < self.free.shape[0] and 0 <= v[1] < self.free.shape[1]) or not self.free[v]:
                    continue
                cost = best[u] + np.hypot(dx, dy)
                if cost < best.get(v, np.inf):
                    best[v], parent[v] = cost, u
                    heapq.heappush(heap, (cost + np.hypot(b[0] - v[0], b[1] - v[1]), v))
        return None

    def subgoal(self, p, goal):
        """The farthest route point visible from p; goal itself if it is visible or there is no route."""
        pts = self.path(p, goal)
        if pts is None:
            return goal
        for q in pts[::-1]:
            if self.sees(p, q):
                return q
        return pts[min(5, len(pts) - 1)]

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
    angles = lidar_angles(len(lidar))     # any ray count (config "lidar_rays")
    pts = p + lidar[:, None] * np.stack([np.cos(yaw + angles), np.sin(yaw + angles)], axis=1)
    clusters, current = [], []
    for k in range(len(lidar)):
        if hit[k] and current and abs(lidar[k] - lidar[current[-1]]) < 0.3:
            current.append(k)
        else:
            if current:
                clusters.append(current)
            current = [k] if hit[k] else []
    if current:
        if clusters and clusters[0][0] == 0 and hit[-1] and abs(lidar[0] - lidar[-1]) < 0.3:
            clusters[0] = current + clusters[0]      # the ring wraps around: the last ray neighbours ray 0
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
    `shorten` then removes detours and needless waits without changing the timing it relies on. Waiting is only considered when something is moving.
    The plan is kept while it stays valid, so the route does not change without a reason; it is replaced when the
    robot strays from it or it runs out. Random numbers are seeded from the time, so runs are reproducible.
    A plan is collision-free for the discs (base + arm) and exact predictions; the real base can still slip off it.
    Parking (last `park_handover` m) is SteerAround's."""
    margins = (0.12, 0.04)         # m of clearance around the robot and arm discs: tried in this order
    # The arm in its driving pose sticks out in front of the base disc. Measured in the base frame, links below the
    # 0.6 m obstacle tops reach x = 0.68, fingertips (higher up) x = 0.74, within y = +-0.05 up to the fingers.
    # These discs cover it (assumes the usual arm start pose, not a scenario's arm_q0).
    arm_offsets, arm_r = np.array([0.46, 0.56, 0.66]), 0.07
    horizon_extra = 25.0           # s the plan may take beyond the straight run
    iters, iters_after_found = 1500, 200
    searches = 4                   # RRT runs per plan, each with its own seed; the best one is kept
    max_reverse = 4.0              # s of reversing allowed in one plan: for short escapes, not as a way to travel
    shorten_iters, wait_trim = 300, 0.5   # tries at shortening the best plan; s removed from a wait at a time
    wait_choices = (1.0, 2.0, 4.0)
    v_plan, w_plan = 0.2, 0.6      # m/s, rad/s of connect's straight run and spin (below the limits, to keep up)
    sample_dt, edge_steps = 0.1, 5 # s between samples; samples per edge
    mover_margin = 0.2             # m kept from movers in plans, more than from static things
    mover_alarm = 0.05             # m: a real gap to a mover below this ends the current plan
    escape_time = 1.5              # s looked ahead when moving away without a plan
    max_stray = 0.3                # m from the plan's reference that triggers a new plan
    check_every = 0.25             # s between plan checks
    park_handover, park_exit = 0.3, 0.6
    via_reached = 0.3              # m from a route point at which the next stretch is planned
    controls = np.array([[0.2, 0.0], [0.2, 0.25], [0.2, -0.25], [0.2, 0.5], [0.2, -0.5], [0.1, 0.5], [0.1, -0.5],
                         [0.0, 0.6], [0.0, -0.6], [-0.1, 0.0], [-0.1, 0.4], [-0.1, -0.4]])

    def __init__(self, robot):
        self.robot = robot
        self.waiting, self.wait_time, self.last_t, self.now = False, 0.0, None, 0.0
        self.plan, self.next_check, self.parking, self.plans, self.plan_failures = None, 0.0, False, 0, 0
        self.spin_dir, self.can_reach, self.next_reach_check, self.escape_motion = 0, False, 0.0, (0.0, 0.0)
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
        self.room, self.box_lo, self.box_hi = r.room, r.box_lo, r.box_hi
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

    def arm_points(self, pts, yaws):
        """Centres of the arm discs for robot positions pts and headings yaws: (len(pts), discs, 2)."""
        heading = np.stack([np.cos(yaws), np.sin(yaws)], axis=1)
        return pts[:, None] + self.arm_offsets[None, :, None]*heading[:, None]

    def clear(self, pts, ts, margin, yaws):
        """Are the base disc and the arm discs clear of the walls and of every obstacle at these times?"""
        arm = self.arm_points(pts, yaws)
        if self.room is not None:
            keep, keep_arm = ROBOT_RADIUS + self.wall_margin, self.arm_r + self.wall_margin
            if np.any(pts < self.room[:2] + keep) or np.any(pts > self.room[2:] - keep):
                return False
            if np.any(arm < self.room[:2] + keep_arm) or np.any(arm > self.room[2:] - keep_arm):
                return False
        fade = np.clip(1 - ts[:, None]/2.0, 0, 1)   # slack only helps the robot get away from where it starts
        if len(self.box_lo):
            if np.any(box_gaps(pts, self.box_lo, self.box_hi) < ROBOT_RADIUS + margin - self.slack_box[None]*fade):
                return False
            if np.any(box_gaps(arm, self.box_lo, self.box_hi).min(axis=1) < self.arm_r + margin - self.slack_box_arm[None]*fade):
                return False
        if len(self.radius) == 0:
            return True
        centres = self.predict(ts)
        gaps = np.linalg.norm(pts[:, None] - centres, axis=2)
        keep = self.obstacle_margin(margin)
        if np.any(gaps < ROBOT_RADIUS + keep + self.radius - self.slack[None]*fade):
            return False
        arm_gaps = np.linalg.norm(arm[:, :, None] - centres[:, None], axis=3).min(axis=1)
        return bool(np.all(arm_gaps >= self.arm_r + keep + self.radius - self.slack_arm[None]*fade))

    def obstacle_margin(self, margin):
        """Clearance per obstacle: a mover always gets at least `mover_margin`, as the base runs a little behind plan."""
        return np.where(self.speed > 0, max(margin, self.mover_margin), margin)

    def mover_too_close(self, p, yaw):
        """Measured from the robot's real pose: is a mover within `mover_alarm` of the base or the arm right now?"""
        self.load_world()
        moving = self.speed > 0
        if not moving.any():
            return False
        here = self.predict(np.zeros(1))[0][moving]
        arm = self.arm_points(p[None], np.array([yaw]))[0]
        return bool(np.any(np.linalg.norm(here - p, axis=1) < ROBOT_RADIUS + self.radius[moving] + self.mover_alarm)
                    or np.any(np.linalg.norm(arm[:, None] - here[None], axis=2).min(axis=0) < self.arm_r + self.radius[moving] + self.mover_alarm))

    def safest_motion(self, p, yaw):
        """With no plan, do not freeze in a mover's way: of the base's motions and standing still, take the one that
        keeps the largest predicted gap to everything over the next `escape_time` seconds. Returns (v, w)."""
        self.load_world()
        ts = np.arange(1, int(self.escape_time/self.sample_dt) + 1)*self.sample_dt
        best, best_gap = (0.0, 0.0), -np.inf
        for v, w in [(0.0, 0.0), *self.controls.tolist(), (-0.2, 0.0), (0.25, 0.0)]:
            th = yaw + w*ts
            if abs(w) > 1e-9:
                xs = p[0] + v/w*(np.sin(th) - np.sin(yaw))
                ys = p[1] - v/w*(np.cos(th) - np.cos(yaw))
            else:
                xs, ys = p[0] + v*ts*np.cos(yaw), p[1] + v*ts*np.sin(yaw)
            pts = np.stack([xs, ys], axis=1)
            arm = self.arm_points(pts, th)
            gaps = [np.inf]
            if self.room is not None:
                gaps += [np.minimum(pts - self.room[:2], self.room[2:] - pts).min() - ROBOT_RADIUS,
                         np.minimum(arm - self.room[:2], self.room[2:] - arm).min() - self.arm_r]
            if len(self.box_lo):
                gaps += [box_gaps(pts, self.box_lo, self.box_hi).min() - ROBOT_RADIUS, box_gaps(arm, self.box_lo, self.box_hi).min() - self.arm_r]
            if len(self.radius):
                centres = self.predict(ts)
                gaps += [(np.linalg.norm(pts[:, None] - centres, axis=2) - self.radius).min() - ROBOT_RADIUS,
                         (np.linalg.norm(arm[:, :, None] - centres[:, None], axis=3) - self.radius).min() - self.arm_r]
            if min(gaps) > best_gap + 1e-3:
                best, best_gap = (v, w), min(gaps)
        return best

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
        if not self.clear(np.stack([xs, ys], axis=1), ts, margin, yaws):
            return None
        vs = np.concatenate([np.zeros(n_spin), np.full(n_run, self.v_plan)])
        ws = np.concatenate([np.full(n_spin, np.sign(turn)*self.w_plan), np.zeros(n_run)])
        return np.stack([ts, xs, ys, yaws, vs, ws], axis=1)

    def search(self, p, yaw, goal, margin, rng):
        """RRT from (p, yaw) at time 0. Returns (rows, reached_goal), or (None, False) if not even one edge is free."""
        here = self.predict(np.zeros(1))[0] if len(self.radius) else np.zeros((0, 2))
        keep = self.obstacle_margin(margin)
        self.slack = np.maximum(ROBOT_RADIUS + keep + self.radius + 0.02 - np.linalg.norm(here - p, axis=1), 0.0)
        arm_now = self.arm_points(p[None], np.array([yaw]))[0]
        arm_gap = np.linalg.norm(arm_now[:, None] - here[None], axis=2).min(axis=0) if len(self.radius) else np.zeros(0)
        self.slack_arm = np.maximum(self.arm_r + keep + self.radius + 0.02 - arm_gap, 0.0)
        if len(self.box_lo):
            self.slack_box = np.maximum(ROBOT_RADIUS + margin + 0.02 - box_gaps(p, self.box_lo, self.box_hi), 0.0)
            self.slack_box_arm = np.maximum(self.arm_r + margin + 0.02 - box_gaps(arm_now, self.box_lo, self.box_hi).min(axis=0), 0.0)
        self.wall_margin = margin
        if self.room is not None:   # never ask for more wall clearance than the start or the goal has
            room_gap = np.minimum(np.minimum(p, goal) - self.room[:2], self.room[2:] - np.maximum(p, goal)).min()
            arm_gap_wall = np.minimum(arm_now - self.room[:2], self.room[2:] - arm_now).min()
            self.wall_margin = float(np.clip(min(room_gap - ROBOT_RADIUS, arm_gap_wall - self.arm_r) - 0.01, 0.0, margin))
        direct = self.connect(*p, yaw, 0.0, goal, margin)
        if direct is not None and not (self.speed > 0).any():
            return direct, True
        straight_time = np.linalg.norm(goal - p)/self.v_plan
        horizon = straight_time + self.horizon_extra
        nodes, parents, edges, reversed_s = [np.array([*p, yaw, 0.0])], [-1], [np.zeros((0, 6))], [0.0]
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
            reversed_s.append(reversed_s[i] + self.sample_dt*np.sum(rows[:, 4] < 0))
            tail = self.connect(*nodes[-1], goal, margin)
            if tail is not None and tail[-1, 0] < best_t:
                best, best_tail, best_t = len(nodes) - 1, tail, tail[-1, 0]

        def wait_from(i, d):
            node = nodes[i]
            ts = node[3] + self.sample_dt*np.arange(1, int(round(d/self.sample_dt)) + 1)
            pts = np.tile(node[:2], (len(ts), 1))
            if ts[-1] <= horizon and self.clear(pts, ts, margin, np.full(len(ts), node[2])):
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
            if reversed_s[i] + self.sample_dt*self.edge_steps > self.max_reverse:
                order = order[self.controls[order, 0] >= 0]
            for k in order[:3]:
                pts = np.stack([ex[k], ey[k]], axis=1)
                if self.clear(pts, ts, margin, th + self.edges[k, :, 2]):
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
            return np.vstack(parts + [best_tail]), True
        if len(nodes) == 1:
            return None, False
        # Goal not reached: keep the plan that stays safe longest, ending as close to the goal as possible
        i = min(range(1, len(nodes)), key=lambda k: (-round(nodes[k][3], 1), np.linalg.norm(nodes[k][:2] - goal)))
        return np.vstack(chain(i)), False

    def shortcut(self, rows, i, j, margin):
        """Replace rows[i+1:j] by a direct move, arriving by rows[j]'s time and with its heading, so everything from
        j on keeps its exact timing and stays valid. Returns the new rows, or None if it is too slow or hits something."""
        _, x, y, th = rows[i, :4]
        seg = self.connect(x, y, th, rows[i, 0], rows[j, 1:3], margin)
        if seg is None or len(seg) == 0:
            return None
        t, th = seg[-1, 0], seg[-1, 3]
        turn = wrap(rows[j, 3] - th)
        n_spin = int(np.ceil(abs(turn)/self.w_plan/self.sample_dt))
        n_wait = int(round((rows[j, 0] - t)/self.sample_dt)) - n_spin - 1
        if n_wait < 0:
            return None
        ts = t + self.sample_dt*np.arange(1, n_wait + n_spin + 1)
        yaws = np.concatenate([np.full(n_wait, th), th + turn*np.arange(1, n_spin + 1)/max(n_spin, 1)])
        fill = np.column_stack([ts, np.tile(rows[j, 1:3], (len(ts), 1)), yaws, np.zeros(len(ts)),
                                np.concatenate([np.zeros(n_wait), np.full(n_spin, np.sign(turn)*self.w_plan)])])
        if len(fill) and not self.clear(fill[:, 1:3], fill[:, 0], margin, yaws):
            return None
        return np.vstack([rows[:i + 1], seg, fill, rows[j:]])

    def trim_wait(self, rows, k, margin):
        """Remove `wait_trim` seconds of the wait containing row k and run everything after it that much earlier.
        Returns the new rows, or None if the earlier timing would hit something."""
        n = int(round(self.wait_trim/self.sample_dt))
        end = k
        while end < len(rows) - 1 and rows[end + 1, 4] == 0 and rows[end + 1, 5] == 0:
            end += 1
        if end - k + 1 < n:
            return None
        tail = rows[k + n:].copy()
        tail[:, 0] -= n*self.sample_dt
        if not self.clear(tail[:, 1:3], tail[:, 0], margin, tail[:, 3]):
            return None
        return np.vstack([rows[:k], tail])

    def shorten(self, rows, p, yaw, margin, rng):
        """Standard RRT clean-up, keeping the plan's timing valid: repeatedly try a shortcut between two random rows
        (shorter path, arrival unchanged) or trim a wait (earlier arrival), keeping only collision-free changes."""
        rows = np.vstack([[0.0, *p, yaw, 0.0, 0.0], rows])
        length = lambda r: np.linalg.norm(np.diff(r[:, 1:3], axis=0), axis=1).sum()
        for _ in range(self.shorten_iters):
            waits = np.flatnonzero((rows[1:, 4] == 0) & (rows[1:, 5] == 0)) + 1
            if len(waits) and rng.random() < 0.5:
                trial = self.trim_wait(rows, int(waits[rng.integers(len(waits))]), margin)
            else:
                i, j = np.sort(rng.integers(0, len(rows), 2))
                trial = self.shortcut(rows, i, j, margin) if j - i >= 5 else None
                if trial is not None and length(trial) >= length(rows) - 1e-6:
                    trial = None
            if trial is not None:
                rows = trial
        return rows[1:]

    def park(self, p, yaw, velocity, target):
        """Turn to face the target, keeping one turning direction until roughly facing it, then creep to the reach
        distance. Starting a turn from close by (or facing away) does not hunt left and right."""
        to_target = target[:2] - p
        err = wrap(np.arctan2(to_target[1], to_target[0]) - yaw)
        radial = np.linalg.norm(to_target) - standoff(target[2])
        moving = abs(np.cos(yaw)*velocity[0] + np.sin(yaw)*velocity[1]) > self.stopped
        if abs(err) < 0.1:
            self.spin_dir = 0
        elif self.spin_dir == 0 and abs(err) > 0.3:
            self.spin_dir = np.sign(err)
        v, w, arrived = 0.0, 0.0, False
        if self.spin_dir != 0:
            if not moving:
                w = self.spin_dir*float(np.clip(3.0*abs(err), 0.2, self.max_spin))
        elif abs(radial) > self.park_tol:
            v = float(np.clip(self.kp_v*radial, -self.creep_v, self.creep_v))
            w = float(np.clip(3.0*err, -self.max_curve, self.max_curve))
        elif abs(err) > self.ang_tol:
            if not moving:
                w = float(np.clip(3.0*err, -self.max_spin, self.max_spin))
        else:
            arrived = not moving
        l, r = (v - w*self.track/2)/self.wheel_radius, (v + w*self.track/2)/self.wheel_radius
        return np.array([r, l, l, r]), arrived

    def reset_for_next_dot(self):
        self.plan, self.parking, self.next_check, self.spin_dir = None, False, self.now, 0
        self.can_reach, self.next_reach_check = False, self.now

    def make_plan(self, p, yaw, goal):
        """Best of `searches` RRT runs with different fixed seeds: the earliest arrival, at the largest margin that
        reaches the goal. Stops early once a plan is about as fast as the straight run."""
        self.load_world()
        self.plans += 1
        straight_time = np.linalg.norm(goal - p)/self.v_plan
        fallback = None
        for margin in self.margins:
            best = None
            for k in range(self.searches):
                rows, reached = self.search(p, yaw, goal, margin, np.random.default_rng([int(self.now*1000), k]))
                if reached and (best is None or rows[-1, 0] < best[-1, 0]):
                    best = rows
                elif rows is not None and not reached and fallback is None:
                    fallback = (rows, margin)
                if best is not None and best[-1, 0] <= 1.05*straight_time:
                    break
            if best is not None:
                seed = np.random.default_rng([int(self.now*1000), self.searches])
                return self.store(self.shorten(best, p, yaw, margin, seed), p, yaw, goal, margin)
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

    def plan_ok(self, p, yaw, goal):
        rows = self.plan["rows"]
        if self.mover_too_close(p, yaw):
            return False
        if self.now >= rows[-1, 0] or np.linalg.norm(rows[-1, 1:3] - goal) > 0.3:
            return False
        return np.linalg.norm(self.ref(self.now)[0][:2] - p) <= self.max_stray

    def wheels(self, p, yaw, velocity, lidar, target):
        to_target = target[:2] - p
        goal = target[:2] - standoff(target[2]) * to_target / max(np.linalg.norm(to_target), 1e-9)
        if self.robot.fixed_park is not None:   # in a floor plan the line to the target may cross a wall
            goal = self.robot.fixed_park
        dist = np.linalg.norm(goal - p)
        if self.robot.choose_spot and self.now >= self.next_reach_check:
            self.next_reach_check = self.now + self.check_every
            self.can_reach = self.robot.can_reach_from(p, yaw)
        self.parking = dist < (self.park_exit if self.parking else self.park_handover) or self.can_reach \
            or (self.parking and abs(np.linalg.norm(to_target) - standoff(target[2])) < 2*REACH_BAND)
        self.waiting = False
        if self.parking:     # the plan brought it here, and spots are chosen away from movers and furniture
            return self.park(p, yaw, velocity, target)
        if self.now >= self.next_check:
            self.next_check = self.now + self.check_every
            if self.plan is None or not self.plan_ok(p, yaw, self.plan["goal"] if self.plan["via"] else goal) \
                    or (self.plan["via"] and np.linalg.norm(self.plan["goal"] - p) < self.via_reached):
                route = self.robot.route
                sub = route.subgoal(p, goal) if route is not None else goal
                self.plan = self.make_plan(p, yaw, sub)
                if self.plan is not None:
                    self.plan["via"] = bool(np.linalg.norm(sub - goal) > 1e-6)
                else:
                    self.escape_motion = self.safest_motion(p, yaw)
        if self.plan is None:   # no plan right now: move to the safest nearby place until there is one
            v, w = self.escape_motion
            l, r = (v - w*self.track/2)/self.wheel_radius, (v + w*self.track/2)/self.wheel_radius
            self.waiting = v == 0 and w == 0
            return np.array([r, l, l, r]), False
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

class VisibleRRTExpert(RRTExpert):
    """RRTExpert limited to what the lidar can show, so the learned base is not taught to react to things it cannot
    sense. A mover counts only if a ray hit it in the last `memory` seconds, and it is predicted at constant velocity,
    without its turning points. Its true position and velocity are used, so this removes hidden knowledge, not
    perception error. Walls, furniture and static cylinders stay known. The current plan is rechecked against what is
    visible now and replaced if a mover blocks it, since a mover can appear or turn after the plan was made."""
    memory = 1.0            # s: about the span of the lidar scans the learned base gets
    recheck_horizon = 8.0   # s of the plan rechecked: constant velocity is not worth trusting further ahead
    recheck_margin = 0.05   # m kept from movers in the recheck; smaller than mover_margin, or plans would churn

    def __init__(self, robot):
        super().__init__(robot)
        self.seen = {}      # obstacle index -> last time a lidar ray hit it

    def observe(self, t, p, yaw, lidar):
        super().observe(t, p, yaw, lidar)
        for k, body in enumerate(self.robot.obstacle_ids):
            if body in self.robot.lidar_bodies:
                self.seen[k] = t

    def load_world(self):
        super().load_world()
        last = np.array([self.seen.get(k, -np.inf) for k in range(len(self.speed))])
        keep = (self.speed == 0) | (self.now - last <= self.memory)
        self.d0, self.start, self.radius = self.d0[keep], self.start[keep], self.radius[keep]
        self.speed, self.s0, self.dir = self.speed[keep], self.s0[keep], self.dir[keep]
        self.sweep = np.tile([-np.inf, np.inf], (len(self.speed), 1))

    def plan_ok(self, p, yaw, goal):
        if not super().plan_ok(p, yaw, goal):
            return False
        rows = self.plan["rows"]
        rows = rows[(rows[:, 0] >= self.now) & (rows[:, 0] <= self.now + self.recheck_horizon)]
        moving = self.speed > 0
        if not len(rows) or not moving.any():
            return True
        centres = self.predict(rows[:, 0] - self.now)[:, moving]
        keep = self.radius[moving] + self.recheck_margin
        arm = self.arm_points(rows[:, 1:3], rows[:, 3])
        return bool(np.all(np.linalg.norm(rows[:, None, 1:3] - centres, axis=2) >= ROBOT_RADIUS + keep)
                    and np.all(np.linalg.norm(arm[:, :, None] - centres[:, None], axis=3) >= self.arm_r + keep))

class SteadyRRTExpert(VisibleRRTExpert):
    """VisibleRRTExpert that settles near-ties the same way every time, so the learned base sees one answer for
    similar scenes instead of a random left or right. Of the plans arriving within `tie` of the fastest, it keeps
    the one passing on the same side of the straight line to the goal as its previous plan; with no previous side,
    the side the robot already faces. A clearly faster route still wins. More searches give more plans to compare."""
    tie = 0.10           # plans up to 10% slower than the fastest count as a tie
    searches = 6
    last_side = 0.0      # +1 left of the line to the goal, -1 right, 0 not known yet

    @staticmethod
    def side(rows, p, goal):
        """Mean signed distance (m) of a plan's points from the straight line p -> goal: + left, - right."""
        d = goal - p
        n = np.linalg.norm(d)
        if n < 1e-6:
            return 0.0
        rel = rows[:, 1:3] - p
        return float(np.mean(d[0]*rel[:, 1] - d[1]*rel[:, 0]) / n)

    def wanted_side(self, p, yaw, goal):
        if self.last_side:
            return self.last_side
        return float(np.sign(wrap(yaw - np.arctan2(goal[1] - p[1], goal[0] - p[0]))))

    def make_plan(self, p, yaw, goal):
        """As RRTExpert.make_plan, but keeps every plan that reaches the goal and picks among the near-ties by side."""
        self.load_world()
        self.plans += 1
        straight_time = np.linalg.norm(goal - p)/self.v_plan
        fallback = None
        for margin in self.margins:
            found = []
            for k in range(self.searches):
                rows, reached = self.search(p, yaw, goal, margin, np.random.default_rng([int(self.now*1000), k]))
                if reached:
                    found.append(rows)
                    if rows[-1, 0] <= 1.05*straight_time:   # about the straight run: nothing to go round
                        break
                elif rows is not None and fallback is None:
                    fallback = (rows, margin)
            if found:
                fastest = min(r[-1, 0] for r in found)
                want = self.wanted_side(p, yaw, goal)
                ties = [r for r in found if r[-1, 0] <= (1 + self.tie)*fastest]
                best = max(ties, key=lambda r: (np.sign(self.side(r, p, goal)) == want, -r[-1, 0]))
                seed = np.random.default_rng([int(self.now*1000), self.searches])
                plan = self.store(self.shorten(best, p, yaw, margin, seed), p, yaw, goal, margin)
                self.last_side = float(np.sign(self.side(plan["rows"], p, goal))) or self.last_side
                return plan
        self.plan_failures += 1
        return self.store(fallback[0], p, yaw, goal, fallback[1]) if fallback else None

class BestRRTExpert(VisibleRRTExpert):
    """VisibleRRTExpert pushed towards the optimal plan, so the plan depends less on the random seed and on when it
    was made (a near-optimal plan from here is about the same as one made a moment ago): 24 RRT runs per plan
    instead of 4, and three times the shortening. Teacher comparison, 3-4 Nov."""
    searches = 24
    shorten_iters = 900

class WindowExpert(VisibleRRTExpert):
    """Dynamic window approach (Fox, Burgard and Thrun 1997) as a teacher a network can copy: no plan in memory and
    no random numbers, so the same situation always gets the same command.

    Every `decide_every` s it drives each (speed, turn) option forward for `horizon` s from the current pose, against
    what VisibleRRTExpert knows (walls, furniture, still obstacles, and movers the lidar hit in the last second at
    constant velocity), for the base disc and the arm discs. Options passing closer than `min_gap` are dropped; the
    rest score progress towards the goal (the route's next visible point in a floor plan), clearance up to
    `clear_cap` and speed. If no option is safe it takes the one keeping the largest gap. Parking is the RRT
    experts' park(). `evade` is True when the movers changed the choice (the best option with movers left out
    differs), the label for the evade/target switch."""
    decide_every = 0.01    # s: one decision per saved row, so every label is fresh
    horizon, step = 3.0, 0.1
    speeds = (-0.1, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25)                       # m/s
    turns = (-0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6)          # rad/s; spinning slides the base, so slow
    # Closer than these in the next `horizon` s is not an option. Movers get more: they are predicted at constant
    # velocity but turn back at the ends of their track, and the base runs a little behind its command (the RRT
    # keeps 0.2 m from movers for the same reasons). First try (5 cm for everything): 13/20 clean, 6 mover hits.
    min_gap, mover_gap = 0.05, 0.25
    clear_cap, mover_cap = 0.6, 1.0   # m: more clearance than this earns nothing extra
    w_progress, w_clear, w_speed = 1.0, 0.8, 0.1

    def __init__(self, robot):
        super().__init__(robot)
        self.options = np.array([(v, w) for v in self.speeds for w in self.turns])
        self.next_decision, self.command, self.evade = 0.0, (0.0, 0.0), False

    def option_paths(self, p, yaw):
        """Positions (options, steps, 2) and headings (options, steps) of every option over the horizon."""
        ts = np.arange(1, int(round(self.horizon/self.step)) + 1)*self.step
        v, w = self.options[:, :1], self.options[:, 1:]
        th = yaw + w*ts
        turning = np.abs(w) > 1e-9
        safe_w = np.where(turning, w, 1.0)
        xs = np.where(turning, p[0] + v/safe_w*(np.sin(th) - np.sin(yaw)), p[0] + v*ts*np.cos(yaw))
        ys = np.where(turning, p[1] - v/safe_w*(np.cos(th) - np.cos(yaw)), p[1] + v*ts*np.sin(yaw))
        return ts, np.stack([xs, ys], axis=2), th

    def gaps(self, ts, pts, th):
        """Smallest gap (m) of each option to the still world and to the movers: two arrays (options,)."""
        arm = self.arm_points(pts.reshape(-1, 2), th.ravel()).reshape(*pts.shape[:2], -1, 2)
        still = np.full(len(pts), np.inf)
        if self.room is not None:
            still = np.minimum(still, (np.minimum(pts - self.room[:2], self.room[2:] - pts).min(axis=(1, 2)) - ROBOT_RADIUS))
            still = np.minimum(still, (np.minimum(arm - self.room[:2], self.room[2:] - arm).min(axis=(1, 2, 3)) - self.arm_r))
        if len(self.box_lo):
            still = np.minimum(still, box_gaps(pts, self.box_lo, self.box_hi).min(axis=(1, 2)) - ROBOT_RADIUS)
            still = np.minimum(still, box_gaps(arm, self.box_lo, self.box_hi).min(axis=(1, 2, 3)) - self.arm_r)
        movers = np.full(len(pts), np.inf)
        if len(self.radius):
            centres = self.predict(ts)                                                    # (steps, obstacles, 2)
            base = np.linalg.norm(pts[:, :, None] - centres[None], axis=3) - self.radius - ROBOT_RADIUS
            arm_gap = (np.linalg.norm(arm[:, :, :, None] - centres[None, :, None], axis=4) - self.radius).min(axis=2) - self.arm_r
            gap = np.minimum(base, arm_gap).min(axis=1)                                   # (options, obstacles)
            moving = self.speed > 0
            if (~moving).any():
                still = np.minimum(still, gap[:, ~moving].min(axis=1))
            if moving.any():
                movers = gap[:, moving].min(axis=1)
        return still, movers

    def choose(self, still, movers, progress):
        """Index of the best option for these gaps (still world, movers); if none is safe, the one furthest from
        being unsafe."""
        ok = (still >= self.min_gap) & (movers >= self.mover_gap)
        if not ok.any():
            return int(np.argmax(np.minimum(still - self.min_gap, movers - self.mover_gap)))
        clear = 0.5*np.minimum(still, self.clear_cap)/self.clear_cap + 0.5*np.minimum(movers, self.mover_cap)/self.mover_cap
        score = (self.w_progress*progress + self.w_clear*clear
                 + self.w_speed*np.maximum(self.options[:, 0], 0)/max(self.speeds))
        return int(np.argmax(np.where(ok, score, -np.inf)))

    def decide(self, p, yaw, goal):
        self.load_world()
        ts, pts, th = self.option_paths(p, yaw)
        still, movers = self.gaps(ts, pts, th)
        start = np.linalg.norm(goal - p)
        progress = (start - np.linalg.norm(goal - pts[:, -1], axis=1))/(max(self.speeds)*self.horizon)
        best = self.choose(still, movers, progress)
        self.evade = bool(np.isfinite(movers).any()) and best != self.choose(still, np.full(len(still), np.inf), progress)
        return tuple(self.options[best])

    def wheels(self, p, yaw, velocity, lidar, target):
        to_target = target[:2] - p
        goal = target[:2] - standoff(target[2]) * to_target / max(np.linalg.norm(to_target), 1e-9)
        if self.robot.fixed_park is not None:
            goal = self.robot.fixed_park
        dist = np.linalg.norm(goal - p)
        if self.robot.choose_spot and self.now >= self.next_reach_check:
            self.next_reach_check = self.now + self.check_every
            self.can_reach = self.robot.can_reach_from(p, yaw)
        self.parking = dist < (self.park_exit if self.parking else self.park_handover) or self.can_reach \
            or (self.parking and abs(np.linalg.norm(to_target) - standoff(target[2])) < 2*REACH_BAND)
        if self.parking:
            self.waiting, self.evade = False, False
            return self.park(p, yaw, velocity, target)
        if self.now >= self.next_decision:
            self.next_decision = self.now + self.decide_every
            route = self.robot.route
            self.command = self.decide(p, yaw, route.subgoal(p, goal) if route is not None else goal)
        v, w = self.command
        self.waiting = v == 0 and w == 0
        l, r = (v - w*self.track/2)/self.wheel_radius, (v + w*self.track/2)/self.wheel_radius
        return np.array([r, l, l, r]), False

class FastWindowExpert(WindowExpert):
    """WindowExpert with a top speed set by the scenario ("top_speed", m/s; 0.25 = WindowExpert's) that turns round
    instead of reversing (5 Nov). WindowExpert scores progress over its 3 s horizon, where turning on the spot gains
    nothing, so with the goal behind it backed all the way there at 0.1 m/s (runs/base-tweaks/corner-window). Here,
    with the goal behind, each option also scores how much it turns towards the goal (w_face). Reversing stays
    allowed (forbidding it while a spin was safe trapped it turning on the spot in tight floor plans; at higher top
    speeds a reverse earns little progress anyway). With the arm folded (scenario arm_q0 =
    the fold pose, inside the base outline) there are no arm discs ahead of the base: RRTExpert's discs model the
    unfolded arm 0.46-0.66 m ahead, which made the folded robot keep clear of a phantom arm."""
    # Defaults = variant F of the 5 Nov tests (21 hard rooms at 0.8 m/s: 12 parked, 2 contact runs; the reverse penalty /
    # facing bonus (C, D: 8-9 parked, 7-8 contacts) and turns up to 1.0 rad/s (E1, E2, F2: more sliding contacts) lost)
    w_face, w_reverse = 0.0, 0.0
    min_gap = 0.05      # m from walls, furniture and still cylinders, plus speed_margin x |speed|
    phantom_arm = False  # True: keep RRTExpert's unfolded-arm discs even with the arm folded (as WindowExpert)
    short_look = True    # still world only over the stopping distance (False: the whole 3 s horizon, as WindowExpert)
    unstick, unstick_progress = True, 0.01   # stuck rule in decide(): spin towards the goal when no safe option progresses
    speed_margin = 0.25  # s: extra clearance from the still world = speed_margin x |speed| (0.25 s -> 0.2 m at 0.8 m/s, about
                         # what the phantom arm added in front, but tied to the real danger: braking and sliding at speed)
    max_turn = 0.6       # rad/s, fastest turn while driving; above 0.6 the turns are 11 even steps (5 Nov: at 0.8 m/s a
                         # 0.6 rad/s turn is a 1.33 m radius; the wheels allow ~1 rad/s at full speed, but the base slides)

    def __init__(self, robot):
        super().__init__(robot)
        top = float(getattr(robot, "top_speed", 0.25))
        self.speeds = (-0.1, 0.0) + tuple(top*k/5 for k in range(1, 6))
        if self.max_turn != 0.6:
            self.turns = tuple(np.linspace(-self.max_turn, self.max_turn, 11))
        self.options = np.array([(v, w) for v in self.speeds for w in self.turns])
        if getattr(robot, "arm_folded", False) and not self.phantom_arm:
            self.arm_offsets = np.array([0.0])      # one disc at the base centre, inside the base disc: no effect

    def gaps(self, ts, pts, th):
        """As WindowExpert.gaps, but the still world (walls, furniture, still cylinders) only counts along the stretch
        the base needs to stop from that speed plus `stop_margin` (at least `min_look` s): it decides again every
        10 ms, so a wall 3 s away at full speed is not a reason to drive slowly. Movers keep the whole horizon."""
        still_full, movers = super().gaps(ts, pts, th)
        extra = self.speed_margin*np.abs(self.options[:, 0])
        if not self.short_look:
            return still_full - extra, movers
        v = np.abs(self.options[:, 0])
        look = np.where(v > 1e-9, (v**2/(2*self.brake) + self.stop_margin)/np.maximum(v, 1e-9), self.horizon)
        n = np.clip(np.ceil(np.maximum(look, self.min_look)/self.step).astype(int), 1, len(ts))
        mask = np.arange(len(ts))[None, :] < n[:, None]                                  # (options, steps)
        arm = self.arm_points(pts.reshape(-1, 2), th.ravel()).reshape(*pts.shape[:2], -1, 2)
        per = np.full(pts.shape[:2], np.inf)
        if self.room is not None:
            per = np.minimum(per, np.minimum(pts - self.room[:2], self.room[2:] - pts).min(axis=2) - ROBOT_RADIUS)
            per = np.minimum(per, np.minimum(arm - self.room[:2], self.room[2:] - arm).min(axis=(2, 3)) - self.arm_r)
        if len(self.box_lo):
            per = np.minimum(per, box_gaps(pts, self.box_lo, self.box_hi).min(axis=2) - ROBOT_RADIUS)
            per = np.minimum(per, box_gaps(arm, self.box_lo, self.box_hi).min(axis=(2, 3)) - self.arm_r)
        if len(self.radius) and (self.speed <= 0).any():
            centres = self.predict(ts)[:, self.speed <= 0]                                # (steps, still, 2)
            base = np.linalg.norm(pts[:, :, None] - centres[None], axis=3) - self.radius[self.speed <= 0] - ROBOT_RADIUS
            arm_gap = (np.linalg.norm(arm[:, :, :, None] - centres[None, :, None], axis=4)
                       - self.radius[self.speed <= 0]).min(axis=2) - self.arm_r
            per = np.minimum(per, np.minimum(base, arm_gap).min(axis=2))
        self.end_gap = per[np.arange(len(per)), n - 1]          # clearance where each option's look ends (for choose)
        return np.where(mask, per, np.inf).min(axis=1) - extra, movers

    def choose(self, still, movers, progress):
        """As WindowExpert.choose, but when NO option is safe (already inside min_gap of something), take the option that
        ENDS furthest from the still world (and clear of movers): get out. WindowExpert takes the largest smallest gap
        along the path, which is the same for every option when the robot is already too close; with speed_margin that
        tie went to spinning on the spot, and the fast expert spun next to a cylinder for the rest of the run (5 Nov)."""
        ok = (still >= self.min_gap) & (movers >= self.mover_gap)
        if ok.any() or getattr(self, "end_gap", None) is None:
            return super().choose(still, movers, progress)
        return int(np.argmax(np.minimum(self.end_gap - self.min_gap, movers - self.mover_gap)))

    brake, stop_margin, min_look = 0.8, 0.4, 1.0   # m/s^2 (assumed braking), m kept beyond the stop, s looked ahead at least

    def decide(self, p, yaw, goal):
        self.load_world()
        ts, pts, th = self.option_paths(p, yaw)
        still, movers = self.gaps(ts, pts, th)
        start = np.linalg.norm(goal - p)
        progress = (start - np.linalg.norm(goal - pts[:, -1], axis=1))/(max(self.speeds)*self.horizon)
        to_goal = goal - pts[:, -1]
        facing = np.cos(np.arctan2(to_goal[:, 1], to_goal[:, 0]) - th[:, -1])     # 1 = facing the goal at the end
        now = np.cos(np.arctan2(*(goal - p)[::-1]) - yaw)
        # Turning round instead of reversing (5 Nov, third version): reversing costs w_reverse, so with the goal behind a
        # spin beats a long reverse; a small gain-only bonus for turning towards a goal BEHIND (w_face) only breaks the
        # tie between spinning and standing still. Earlier, larger facing bonuses made it dither: it pulled the robot back
        # whenever it arced away round a cylinder, more so at higher top speeds where progress is scaled down
        turn_bonus = np.maximum(facing - now, 0.0) if now < 0 else np.zeros_like(facing)
        score = (progress + self.w_face*turn_bonus/self.w_progress
                 - self.w_reverse*(self.options[:, 0] < 0)/self.w_progress)
        best = self.choose(still, movers, score)
        if self.unstick:
            # Stuck rule (5 Nov): if no safe option makes progress (the goal behind, the way back blocked), turn on the spot
            # towards the goal instead of standing still. The fast expert drives closer to walls than WindowExpert (it
            # looks only as far as it needs to brake), and in 14 of 120 rooms it then stood still facing away for good
            ok = (still >= self.min_gap) & (movers >= self.mover_gap)
            if ok.any() and progress[ok].max() <= self.unstick_progress:
                side = np.sign(wrap(np.arctan2(*(goal - p)[::-1]) - yaw))         # + = turn left towards the goal
                spin = ok & (self.options[:, 0] == 0) & (np.sign(self.options[:, 1]) == (side or 1.0))
                if spin.any():
                    best = int(np.flatnonzero(spin)[np.argmax(np.abs(self.options[spin, 1]))])
        self.evade = bool(np.isfinite(movers).any()) and best != self.choose(still, np.full(len(still), np.inf), score)
        return tuple(self.options[best])


class ObstacleRobot(Robot):
    def __init__(self, config, scenario, gui=False):
        super().__init__(config, scenario, gui)
        p, client = self.p, self.client
        self.lidar_rays = int(config.get("lidar_rays", LIDAR_RAYS))   # 36 unless the run's config says otherwise
        self.lidar_range = float(config.get("lidar_range", LIDAR_RANGE))   # m; 4 unless the config says otherwise (5 Nov)
        # robot_env made the first dot's marker last and keeps no id: take it now, before any other body is added
        first_dot = client.getBodyUniqueId(client.getNumBodies() - 1)
        if np.linalg.norm(np.array(client.getBasePositionAndOrientation(first_dot)[0]) - self.target) > 1e-6:
            raise RuntimeError("could not identify the first dot's marker")
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
        # Floor plan: axis-aligned boxes {"min": [x, y], "max": [x, y], "kind": "wall" | "furniture", "h": height}.
        # Interior walls show as flat lines like the room boundary; furniture shows as a solid box. Both count as walls
        # for contact. fixed_park: choose the parking spot on arrival (see leg), since the line to the target may cross a wall.
        boxes = scenario.get("boxes", [])
        self.box_lo = np.array([b["min"] for b in boxes], dtype=float).reshape(-1, 2)
        self.box_hi = np.array([b["max"] for b in boxes], dtype=float).reshape(-1, 2)
        for b in boxes:
            lo, hi = np.array(b["min"], dtype=float), np.array(b["max"], dtype=float)
            h = b.get("h", WALL_HEIGHT)
            col = client.createCollisionShape(p.GEOM_BOX, halfExtents=[*((hi - lo)/2), h/2])
            furniture = b.get("kind") == "furniture"
            vis = -1
            if furniture:
                vis = client.createVisualShape(p.GEOM_BOX, halfExtents=[*((hi - lo)/2), h/2], rgbaColor=[0.55, 0.35, 0.2, 1])
            self.wall_ids.append(client.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                                        basePosition=[*((lo + hi)/2), h/2]))
            if not furniture:
                line = client.createVisualShape(p.GEOM_BOX, halfExtents=[*((hi - lo)/2), 0.005], rgbaColor=[0.9, 0.2, 0.1, 1])
                client.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=line,
                                       basePosition=[*((lo + hi)/2), 0.005])
        self.fixed_park = np.array(scenario["park"], dtype=float) if scenario.get("fixed_park") else None
        self.choose_spot, self.fallback_park = bool(scenario.get("fixed_park")), self.fixed_park
        self.route = None
        if boxes and self.room is not None:
            static = [o for o in scenario["obstacles"] if "vel" not in o]
            self.route = FloorPlanRoute(self.room, self.box_lo, self.box_hi, [np.array(o["xy"]) for o in static],
                                        [o["r"] for o in static])
            walls = [b for b in boxes if b.get("kind") != "furniture"]
            self.route.set_rooms(np.array([b["min"] for b in walls], dtype=float).reshape(-1, 2),
                                 np.array([b["max"] for b in walls], dtype=float).reshape(-1, 2))
        # Several dots: scenario["targets"] and, in floor plans, scenario["parks"]. The hand touches a dot (the usual
        # success rule), the dot disappears, the arm returns to its driving pose, and the base goes to the next one.
        self.dots = [np.asarray(t, dtype=float) for t in scenario.get("targets", [scenario["target"]])]
        self.parks = [np.array(q, dtype=float) for q in scenario["parks"]] if "parks" in scenario else None
        self.dot_order = scenario.get("dot_order", "best")
        self.dot_bodies = {}
        if len(self.dots) > 1:
            self.dot_bodies[0] = first_dot
            shape = client.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 0.85])
            for k, dot in enumerate(self.dots[1:], start=1):
                self.dot_bodies[k] = client.createMultiBody(baseMass=0, baseVisualShapeIndex=shape, basePosition=dot.tolist())
        self.remaining, self.picked, self.current_dot = list(range(len(self.dots))), [], 0
        self.retracting, self.leg_start, self.q_hold = False, 0.0, None
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
        # "expert_unfold" (5 Nov): with a folded arm, the expert unfolds it to ARM_START after parking, then reaches
        self.unfold_to = np.array(ARM_START) if scenario.get("expert_unfold") and "arm_q0" in scenario else None
        self.q_unfold = None
        self.top_speed = float(scenario.get("top_speed", 0.25))      # FastWindowExpert's top speed (m/s)
        self.arm_folded = "arm_q0" in scenario and np.allclose(scenario["arm_q0"], [3.14159, 2.618, 3.14159, 0.524, 0.0, 1.309, 0.0], atol=1e-3)
        # sensor-only either way: never sees scenario["obstacles"]
        planner = {"rrt": RRTExpert, "rrt_visible": VisibleRRTExpert, "rrt_steady": SteadyRRTExpert,
                   "rrt_best": BestRRTExpert, "window": WindowExpert,
                   "window_fast": FastWindowExpert}.get(scenario.get("expert"))
        self.base_expert = planner(self) if planner else SteerAround()
        self.base_params.nav_timeout = RRT_NAV_TIMEOUT if planner else NAV_TIMEOUT
        self.arrived = False
        self.choose_dot(np.array([x, y]))

    def spot_ok(self, spot, k, room=0.2):
        """Can the base stand at `spot` to reach dot k? Room for the base (`room` m of margin), a clear line for the arm,
        away from static obstacles and from the whole sweep of every mover."""
        tg = self.dots[k]
        line = spot + np.linspace(0, 1, 8)[:, None]*(tg[:2] - spot)
        if self.room is not None and (np.any(spot < self.room[:2] + ROBOT_RADIUS + room) or np.any(spot > self.room[2:] - ROBOT_RADIUS - room)):
            return False
        if len(self.box_lo) and (box_gaps(spot, self.box_lo, self.box_hi).min() < ROBOT_RADIUS + room
                                 or box_gaps(line, self.box_lo, self.box_hi).min() < 0.2):
            return False
        for c, r, d, (lo, hi) in zip(self.obstacle_start, self.obstacle_r, self.obstacle_vel0, self.obstacle_sweep):
            moving = np.any(d != 0)
            along = np.linspace(max(lo, -9), min(hi, 9), 60) if moving else np.zeros(1)
            path = c + along[:, None]*(d/max(np.linalg.norm(d), 1e-9))
            if np.linalg.norm(path - spot, axis=1).min() < r + ROBOT_RADIUS + (0.35 if moving else room) \
                    or np.linalg.norm(path[:, None] - line[None], axis=2).min() < r + 0.15:
                return False
        return True

    def spots(self, k):
        """Valid parking spots of dot k: points at reach distance from it, on any side."""
        tg = self.dots[k]
        candidates = [tg[:2] - standoff(tg[2])*np.array([np.cos(a), np.sin(a)]) for a in np.arange(0, 2*np.pi, np.pi/18)]
        return [q for q in candidates if self.spot_ok(q, k)]

    def can_reach_from(self, p, yaw):
        """The base is already at a good distance from the current dot with a clear line, so it can stop here. If it
        must still turn a long way, the arm also needs room to swing round."""
        to_target = self.target[:2] - p
        if abs(np.linalg.norm(to_target) - standoff(self.target[2])) > REACH_BAND or not self.spot_ok(p, self.current_dot, room=0.1):
            return False
        facing = abs(wrap(np.arctan2(to_target[1], to_target[0]) - yaw)) < 0.5
        return facing or not len(self.box_lo) or box_gaps(p, self.box_lo, self.box_hi).min() >= 0.85

    def crosses_mover(self, a, spot):
        """Does the route from a to spot cross the sweep of a moving obstacle?"""
        pts = self.route.path(a, spot) if self.route is not None else None
        pts = np.vstack([a, *(pts if pts is not None else [spot])])
        steps = np.vstack([p + np.linspace(0, 1, max(2, int(np.linalg.norm(q - p)/0.1) + 1))[:, None]*(q - p) for p, q in zip(pts[:-1], pts[1:])])
        for c, r, d, (lo, hi) in zip(self.obstacle_start, self.obstacle_r, self.obstacle_vel0, self.obstacle_sweep):
            if np.any(d != 0):
                sweep = c + np.linspace(max(lo, -9), min(hi, 9), 60)[:, None]*(d/np.linalg.norm(d))
                if np.linalg.norm(steps[:, None] - sweep[None], axis=2).min() < r + ROBOT_RADIUS:
                    return True
        return False

    def leg(self, a, k):
        """Cost of the base's way from point a to the best parking spot of dot k, and that spot. In a floor plan the spot
        on the shortest route is chosen, so the robot does not drive around a dot to reach a far side. Crossing a
        mover's sweep costs extra, so among similar routes the one clear of movers is preferred."""
        if not self.choose_spot:
            spot = self.parks[k] if self.parks is not None else \
                self.dots[k][:2] - standoff(self.dots[k][2])*(self.dots[k][:2] - a)/max(np.linalg.norm(self.dots[k][:2] - a), 1e-9)
            return float(np.linalg.norm(spot - a)), spot
        spots = self.spots(k) or [self.parks[k] if self.parks is not None else self.fallback_park]
        if self.route is None:
            best = min(spots, key=lambda q: np.linalg.norm(q - a))
            return float(np.linalg.norm(best - a)), best
        field = self.route.lengths_from(a)
        lengths = [field[self.route.to_cell(q)] for q in spots]
        i = int(np.argmin(lengths))
        length = float(lengths[i]) if np.isfinite(lengths[i]) else float(np.linalg.norm(spots[i] - a)) + 100.0
        return length + (MOVER_CROSS_COST if self.crosses_mover(a, spots[i]) else 0.0), spots[i]

    def choose_dot(self, p):
        """Make the next dot current. "nearest" takes the closest. "best" compares every order of the remaining dots
        (fine up to about 6): route length plus a penalty per room change, so the robot finishes a room before leaving
        it. Among orders within `ORDER_TIE` of the best, it takes the fewest room changes, then the closest dot first, so it
        does not walk past a dot to save a few centimetres. Recomputed after each pick."""
        if len(self.dots) == 1:
            if self.choose_spot:
                self.fixed_park = self.leg(p, 0)[1]
            return
        if self.dot_order == "nearest" or len(self.remaining) == 1:
            k = min(self.remaining, key=lambda i: self.leg(p, i)[0])
        else:
            def score(order):
                a, cost, changes = p, 0.0, 0
                for i in order:
                    d, spot = self.leg(a, i)
                    cost += d
                    if self.route is not None and self.route.room_of(a) != self.route.room_of(spot):
                        changes += 1
                    a = spot
                return cost + ROOM_CHANGE_COST*changes, changes
            scores = {order: score(order) for order in itertools.permutations(self.remaining)}
            best = min(c for c, _ in scores.values())
            close = [order for order, (c, _) in scores.items() if c <= best*(1 + ORDER_TIE)]
            k = min(close, key=lambda order: (scores[order][1], self.leg(p, order[0])[0]))[0]
            if self.route is not None:   # within the room the order goes to next, take its closest dot first
                room = self.route.room_of(self.leg(p, k)[1])
                k = min((i for i in self.remaining if self.route.room_of(self.leg(p, i)[1]) == room), key=lambda i: self.leg(p, i)[0])
        self.current_dot, self.target = k, self.dots[k]
        if self.choose_spot:
            self.fixed_park = self.leg(p, k)[1]

    def dot_touched(self, state, elapsed):
        """The hand touched the current dot. Returns True if it was the last one; otherwise starts the arm's return."""
        k = self.current_dot
        self.picked.append({"dot": k, "time_s": round(float(elapsed), 3)})
        self.remaining.remove(k)
        if k in self.dot_bodies:
            self.client.removeBody(self.dot_bodies.pop(k))
        if not self.remaining:
            return True
        self.choose_dot(state["base"][:2])
        self.retracting, self.q_hold = True, state["q"][4:11].copy()
        return False

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
        """lidar_rays (default 36) distances (m, from the base centre) to the nearest obstacle in each direction;
        lidar_range (default LIDAR_RANGE = 4 m) if none."""
        base, yaw = state["base"], state["yaw"]
        angles = lidar_angles(self.lidar_rays)
        u = np.stack([np.cos(yaw + angles), np.sin(yaw + angles), np.zeros(self.lidar_rays)], axis=1)
        origin = np.array([base[0], base[1], base[2] + LIDAR_HEIGHT])
        reach = self.lidar_range
        hits = self.client.rayTestBatch((origin + LIDAR_START*u).tolist(), (origin + reach*u).tolist())
        self.lidar_bodies = {h[0] for h in hits}
        return np.array([LIDAR_START + h[2]*(reach - LIDAR_START) if h[0] >= 0 else reach for h in hits])

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
        if self.retracting:
            # Ramp the hold pose back: the controller's hold is a joint-space PD, so a jump would swing the arm hard
            step = np.clip(self.q_init - self.q_hold, -RETRACT_RATE*self.control_dt, RETRACT_RATE*self.control_dt)
            self.q_hold = self.q_hold + step
            torque, self.q_des, _ = self.expert.arm_controller(self.dyn, state["q_pin"], state["q"][4:11], state["qd"][4:11],
                self.q_des, self.q_hold, self.target, state["ee"], self.ee_id, self.shoulder_id, self.lower, self.upper,
                self.control_dt, self.arm_params, reach_active=False)
            # Done when the ramp has ended and the arm has settled; it settles a little off q_init (about 0.13 rad at
            # joint 6), as the hold's PD and gravity compensation leave a small steady error
            if np.abs(self.q_hold - self.q_init).max() < 1e-9 and np.abs(state["qd"][4:11]).max() < RETRACT_SETTLED:
                self.retracting, self.arrived, self.leg_start = False, False, elapsed
                self.q_des = self.q_init.copy()
                if hasattr(self.base_expert, "reset_for_next_dot"):
                    self.base_expert.reset_for_next_dot()
            return np.concatenate([np.zeros(4), torque])
        if self.arrived and self.unfold_to is not None:
            # Unfold before reaching: ramp the hold pose from the folded pose to ARM_START (as the learned robot does)
            if self.q_unfold is None:
                self.q_unfold = self.q_init.copy()
                # The straight path swings the arm out to the right (to y = -0.54 m); going round the other way
                # (joints 1 and 3 to 2 pi, the same pose) swings it to the left (+0.55 m). Take the side with more room.
                a, lid = lidar_angles(len(state["lidar"])), np.asarray(state["lidar"])
                left = lid[(a >= 0.9) & (a <= 2.2)].min()
                right = lid[(a >= 2*np.pi - 2.2) & (a <= 2*np.pi - 0.9)].min()
                if left > right:
                    self.unfold_to = self.unfold_to + np.array([2*np.pi, 0, 2*np.pi, 0, 0, 0, 0])
            step = np.clip(self.unfold_to - self.q_unfold, -RETRACT_RATE*self.control_dt, RETRACT_RATE*self.control_dt)
            self.q_unfold = self.q_unfold + step
            torque, self.q_des, _ = self.expert.arm_controller(self.dyn, state["q_pin"], state["q"][4:11], state["qd"][4:11],
                self.q_des, self.q_unfold, self.target, state["ee"], self.ee_id, self.shoulder_id, self.lower, self.upper,
                self.control_dt, self.arm_params, reach_active=False)
            if np.abs(self.q_unfold - self.unfold_to).max() < 1e-9 and np.abs(state["qd"][4:11]).max() < RETRACT_SETTLED:
                self.q_init, self.unfold_to = self.unfold_to.copy(), None
                self.q_des = self.q_init.copy()
            return np.concatenate([np.zeros(4), torque])
        if not self.arrived:
            wheel, self.arrived = self.base_expert.wheels(state["base"][:2], state["yaw"], state["x"][22:24],
                                                          state["lidar"], self.target)
            if elapsed - self.leg_start > self.base_params.nav_timeout:
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
            perturb=None, label_arm=False, contact_limit_s=None, stall_limit_s=None):
    """Like robot_env.rollout, plus lidar, collisions and base arrival. policy.act(state) -> 11 commands.

    label_expert (DAgger): while a learned policy drives, also ask the base expert what wheel speeds it would use
    in each visited state and save them as "expert_wheels". They never move the robot. SteerAround is memoryless and
    sensor-only, so it can label any state from the current sensors alone. An RRT expert (scenario["expert"]) has a
    plan, so it runs as a SHADOW: it gets the clock and sensors every step, plans from wherever the policy took the
    robot, and replans when the robot strays from that plan.
    contact_limit_s, stall_limit_s (DAgger): end the episode once the robot has been in contact this long ("crashed")
    or has barely moved this long ("stalled"), so stuck runs do not fill the data.
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
                   + (["expert_arm"] if label_arm else []) + (["obstacle_xy", "obstacle_seen"] if robot.has_moving_obstacles else []) \
                   + (["contact_obstacle"] if robot.has_moving_obstacles else []) \
                   + (["waiting"] if hasattr(robot.base_expert, "waiting") else []) \
                   + (["expert_evade"] if hasattr(robot.base_expert, "evade") else []) \
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
            touching, still = 0.0, 0.0
            arrival_time = None
            min_clearance = robot.clearance()
            state = robot.measure()
            while elapsed < config["duration_s"]:
                start = time.perf_counter()
                seen = state
                if noisy_lidar:
                    lid = state["lidar"] + rng.normal(0, pert.get("lidar_noise", 0), robot.lidar_rays)
                    lid[rng.random(robot.lidar_rays) < pert.get("lidar_dropout", 0)] = robot.lidar_range
                    seen = {**state, "lidar": np.clip(lid, LIDAR_START, robot.lidar_range)}
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
                    rows["obstacle_seen"].append([b in robot.lidar_bodies for b in robot.obstacle_ids])
                    rows["contact_obstacle"].append(hit.copy())
                if "contact_wall" in rows:
                    rows["contact_wall"].append(robot.touching_wall())
                if "waiting" in rows:
                    rows["waiting"].append(robot.base_expert.waiting)
                if label_expert:
                    if hasattr(robot.base_expert, "observe"):   # RRT: the shadow needs the clock and what the lidar hit
                        robot.base_expert.observe(elapsed, state["base"][:2], state["yaw"], state["lidar"])
                    label, parked = robot.base_expert.wheels(state["base"][:2], state["yaw"], state["x"][22:24],
                                                             state["lidar"], robot.target)
                    rows["expert_wheels"].append(np.zeros(4) if parked else label)
                if "expert_evade" in rows:   # after the label, so in DAgger it belongs to this state
                    rows["expert_evade"].append(robot.base_expert.evade)
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
                    if robot.dot_touched(state, elapsed):
                        reason = "success"
                        break
                    dwell = 0.0
                    state = robot.measure()   # the current dot changed
                if lost_distance and np.linalg.norm(robot.target[:2] - state["base"][:2]) > lost_distance:
                    reason = "lost"
                    break
                touching = touching + robot.control_dt if rows["contact"][-1] else 0.0
                still = still + robot.control_dt if np.linalg.norm(state["x"][22:24]) < STILL_SPEED else 0.0
                if contact_limit_s and touching >= contact_limit_s:
                    reason = "crashed"
                    break
                if stall_limit_s and still >= stall_limit_s:
                    reason = "stalled"
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
                "contact_steps": contact_steps, "dots_total": len(robot.dots), "dots_picked": len(robot.picked),
                "picks": robot.picked,
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
                "lidar": {"rays": robot.lidar_rays, "start_m": LIDAR_START, "range_m": robot.lidar_range, "height_m": LIDAR_HEIGHT},
                "obstacle_height_m": OBSTACLE_HEIGHT, "wheel_side_grip": WHEEL_SIDE_GRIP,
                "saved_every": save_every}, indent=2))
        return summary
    except Exception as error:
        (folder/"failure.json").write_text(json.dumps({"error": repr(error), "scenario": scenario}, indent=2))
        raise
    finally:
        if robot is not None:
            robot.close()
