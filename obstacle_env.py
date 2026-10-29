"""Obstacle world: static cylinders, a random start pose, a 2D lidar and a steer-around expert for the base.

Builds on robot_env.Robot (unchanged) and the course arm expert. The base expert is ours:
a memoryless "steer-around" (vortex) field, so its wheel command depends only on what it senses right now,
never on an internal mode. It is sensor-only: lidar, own pose/velocity and the target position, exactly what
a learned policy gets. Obstacle positions are used only to build the world and to count contacts (referee).
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

    def wheels(self, p, yaw, velocity, lidar, target):
        """Wheel speeds [FR, FL, BL, BR] in rad/s, and whether the base is parked facing the target."""
        to_target = target[:2] - p
        r = np.linalg.norm(to_target)
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
                w = float(np.clip(1.5*err, -self.max_curve, self.max_curve))
        left = (v - w*self.track/2) / self.wheel_radius
        right = (v + w*self.track/2) / self.wheel_radius
        return np.array([right, left, left, right]), arrived

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
        for o in scenario["obstacles"]:
            col = client.createCollisionShape(p.GEOM_CYLINDER, radius=o["r"], height=OBSTACLE_HEIGHT)
            vis = client.createVisualShape(p.GEOM_CYLINDER, radius=o["r"], length=OBSTACLE_HEIGHT,
                                           rgbaColor=[0.45, 0.45, 0.5, 1])
            self.obstacle_ids.append(client.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis, basePosition=[o["xy"][0], o["xy"][1], OBSTACLE_HEIGHT/2]))
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
        self.base_expert = SteerAround()   # sensor-only: never sees scenario["obstacles"]
        self.base_params.nav_timeout = NAV_TIMEOUT
        self.arrived = False

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

    def touching(self):
        return any(self.client.getContactPoints(self.rid, o) for o in self.obstacle_ids)

    def clearance(self, up_to=2.0):
        """Referee only: gap (m) between the robot's body (any link) and the nearest obstacle surface."""
        gaps = [c[8] for o in self.obstacle_ids for c in self.client.getClosestPoints(self.rid, o, up_to)]
        return float(min(gaps, default=up_to))

    def expert_action(self, state, elapsed):
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
                   + (["expert_arm"] if label_arm else [])
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
                for k, v in {"observations": state["x"], "lidar": state["lidar"], "actions": action,
                             "raw_actions": raw, "time_s": elapsed, "ee_m": state["ee"], "error_m": state["error"],
                             "base_m": state["base"], "inference_ms": latency, "arm_q_rad": state["q"][4:11],
                             "contact": robot.touching()}.items():
                    rows[k].append(np.copy(v))
                if noisy_lidar:
                    rows["lidar_seen"].append(np.copy(seen["lidar"]))
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
                "contact_steps": contact_steps, "contact_fraction": contact_steps/len(arrays["time_s"]),
                "min_lidar_m": float(arrays["lidar"].min()), "min_clearance_m": min_clearance,
                "base_path_length_m": float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum()),
                "clipped_action_fraction": clipped/len(arrays["time_s"]),
                "base_nav_timeout_triggered": bool(policy is None and robot.base_timed_out),
                "control_period_s": robot.control_dt, "samples": len(arrays["time_s"]), "perturb": pert}
            np.savez_compressed(folder/"trajectory.npz", **{k: v[::save_every] for k, v in arrays.items()})
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
