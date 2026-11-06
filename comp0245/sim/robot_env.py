"""Adapter around the course expert and RoboEnv; external repositories stay untouched."""

import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from comp0245.models import History, clip_action, observation


def sha256(path):
    """SHA-256 hex digest of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(config):
    """Hashes of the course files the expert comes from, so each episode records the exact
    source."""
    root = Path(config["roboenv_root"])
    task = Path(config["course_root"]) / "final/robotnik_arm_task"
    return {
        str(p): sha256(p)
        for p in [
            root / "configs/robotnik_arm.json",
            task / "robotnik_arm_control.py",
            task / "robotnik_arm_task.py",
        ]
    }


class Robot:
    """The course robot in PyBullet with the course expert controller (no obstacles)."""

    def __init__(self, config, scenario, gui=False):
        """Start the simulator, load the robot and target marker, and set up the expert.

        The base goal is a standoff point in front of the target (distance depends on target
        height), facing it.
        """
        import pinocchio as pin
        import pybullet as p
        from simulation_and_control import MotorCommands, PinWrapper, pb

        root = Path(config["roboenv_root"]).resolve()
        task = Path(config["course_root"]).resolve() / "final/robotnik_arm_task"
        sys.path.insert(0, str(task))
        import robotnik_arm_control as expert

        self.pin, self.p, self.expert, self.config = pin, p, expert, config
        self.target = np.asarray(scenario["target"], dtype=float)
        np.random.seed(scenario["seed"])
        self.sim = pb.SimInterface("robotnik_arm.json", conf_file_path_ext=str(root), use_gui=gui)
        self.client, self.rid = self.sim.pybullet_client, self.sim.bot[0].bot_pybullet
        self.sim.SetFloorFriction(1.0)
        for j in range(self.client.getNumJoints(self.rid)):
            if "wheel" in self.client.getJointInfo(self.rid, j)[1].decode():
                self.client.changeDynamics(
                    self.rid,
                    j,
                    contactStiffness=30000,
                    contactDamping=1000,
                    anisotropicFriction=[1.0, 0.01, 1.0],
                )
        self.sim.bot[0].servo_motor_model.set_motor_gains(400, 30)
        expert.configure_gripper_fingers(self.sim, self.client, self.rid, np.full(6, 0.1))
        shape = self.client.createVisualShape(
            p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 0.85]
        )
        self.client.createMultiBody(
            baseMass=0, baseVisualShapeIndex=shape, basePosition=self.target
        )
        names = np.expand_dims(np.array(self.sim.getNameActiveJoints()), axis=0)
        self.dyn = PinWrapper(
            "robotnik_arm.json", "pybullet", names, ["pybullet"], False, 0, str(root)
        )
        self.ee_id = self.dyn.pin_model.getFrameId("robot_j2s7s300_end_effector")
        self.shoulder_id = self.dyn.pin_model.getFrameId("robot_j2s7s300_link_2")
        self.dt = self.sim.GetTimeStep()
        self.interval = config["physics_steps_per_action"]
        self.control_dt = self.dt * self.interval
        self.q_init = np.array(self.sim.GetMotorAngles(0))[4:11].copy()
        self.q_des = self.q_init.copy()
        self.lower, self.upper = [np.asarray(x)[4:11] for x in self.sim.GetBotJointsLimit()]
        self.base_state = expert.BaseNavState()
        self.base_timed_out = False
        self.base_params = expert.BaseControllerParams(nav_timeout=25.0)
        self.arm_params = expert.ArmControllerParams(reach_limit=0.85)
        standoff = np.interp(self.target[2], [0.1, 0.5, 1.1, 1.25], [0.80, 0.65, 0.65, 0.35])
        bearing = (
            np.arctan2(self.target[1], self.target[0])
            if np.linalg.norm(self.target[:2]) > 1e-4
            else 0.0
        )
        self.goal = self.target[:2] - standoff * np.array([np.cos(bearing), np.sin(bearing)])
        self.goal_bearing = bearing
        self.command = MotorCommands()

    def measure(self):
        """Read the current state.

        Returns a dict: x (28,) observation, base position (3,) m, yaw rad, all joint angles q and
        speeds qd, the pinocchio configuration, hand position ee (3,) m and its distance to the
        target (m).
        """
        base = np.asarray(self.sim.GetBasePosition())
        ori = np.asarray(self.sim.GetBaseOrientation())
        yaw = self.expert.quaternion2bearing(ori[3], *ori[:3])
        q, qd = np.asarray(self.sim.GetMotorAngles(0)), np.asarray(self.sim.GetMotorVelocities(0))
        q_pin = self.dyn.ReoderJoints2PinVec(np.concatenate([base, ori, q]), "pos")
        self.pin.forwardKinematics(self.dyn.pin_model, self.dyn.pin_data, q_pin)
        self.pin.updateFramePlacements(self.dyn.pin_model, self.dyn.pin_data)
        ee = self.dyn.pin_data.oMf[self.ee_id].translation.copy()
        x = observation(
            base,
            yaw,
            self.target,
            q[4:11],
            qd[4:11],
            self.sim.GetBaseLinVelocity(),
            self.sim.GetBaseAngVelocity(),
        )
        return {
            "x": x,
            "base": base,
            "yaw": yaw,
            "q": q,
            "qd": qd,
            "q_pin": q_pin,
            "ee": ee,
            "error": float(np.linalg.norm(ee - self.target)),
        }

    def expert_action(self, state, elapsed):
        """The expert's command for this state: (11,) = 4 wheel speeds (rad/s) then 7 arm torques
        (N m)."""
        was_done = self.base_state.done
        wheel = self.expert.base_controller(
            self.base_state,
            state["base"],
            state["yaw"],
            self.goal,
            self.goal_bearing,
            elapsed,
            self.base_params,
        )
        if not was_done and self.base_state.done and elapsed > self.base_params.nav_timeout:
            self.base_timed_out = True
        torque, self.q_des, _ = self.expert.arm_controller(
            self.dyn,
            state["q_pin"],
            state["q"][4:11],
            state["qd"][4:11],
            self.q_des,
            self.q_init,
            self.target,
            state["ee"],
            self.ee_id,
            self.shoulder_id,
            self.lower,
            self.upper,
            self.control_dt,
            self.arm_params,
            reach_active=self.base_state.done,
        )
        return np.concatenate([wheel, torque])

    def step(self, action):
        """Apply one command (11,) for one control interval (physics_steps_per_action steps)."""
        self.command.SetControlCmd(
            np.concatenate([action, np.zeros(6)]), ["velocity"] * 4 + ["torque"] * 13
        )
        for _ in range(self.interval):
            self.sim.Step(self.command, "torque")

    def close(self):
        """Disconnect the simulator."""
        self.client.disconnect()


def rollout(config, scenario, folder, policy=None, gui=False):
    """Every row is observation BEFORE its corresponding applied action. Final state is
    separate."""
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(f"Refusing to overwrite episode: {folder}")
    folder.mkdir(parents=True)
    robot = None
    try:
        with (folder / "simulator.log").open("w") as log, contextlib.redirect_stdout(log):
            robot = Robot(config, scenario, gui)
            history = History(config["history"])
            rows = {
                key: []
                for key in [
                    "observations",
                    "actions",
                    "raw_actions",
                    "time_s",
                    "ee_m",
                    "error_m",
                    "base_m",
                    "inference_ms",
                    "arm_q_rad",
                ]
            }
            elapsed, dwell, clipped = 0.0, 0.0, 0
            reason = "timeout"
            state = robot.measure()
            while elapsed < config["duration_s"]:
                features = history.append(state["x"])
                start = time.perf_counter()
                # A learned policy never calls the expert or receives its internal state.
                raw = (
                    robot.expert_action(state, elapsed)
                    if policy is None
                    else policy.predict(features[None])[0]
                )
                latency = (time.perf_counter() - start) * 1000
                action = clip_action(raw, config)
                clipped += int(np.any(np.abs(raw - action) > 1e-10))
                for key, value in {
                    "observations": state["x"],
                    "actions": action,
                    "raw_actions": raw,
                    "time_s": elapsed,
                    "ee_m": state["ee"],
                    "error_m": state["error"],
                    "base_m": state["base"],
                    "inference_ms": latency,
                    "arm_q_rad": state["q"][4:11],
                }.items():
                    rows[key].append(np.copy(value))
                robot.step(action)
                elapsed += robot.control_dt
                state = robot.measure()
                dwell = (
                    dwell + robot.control_dt
                    if state["error"] <= config["success_tolerance_m"]
                    else 0.0
                )
                if elapsed >= config["min_success_time_s"] and dwell >= config["success_hold_s"]:
                    reason = "success"
                    break
            arrays = {k: np.asarray(v) for k, v in rows.items()}
            latency = arrays["inference_ms"]
            # Include terminal state in path and error measurements, but not as an action-labelled
            # row.
            path = np.vstack([arrays["base_m"], state["base"]])
            arm = np.vstack([arrays["arm_q_rad"], state["q"][4:11]])
            summary = {
                "scenario_id": scenario["id"],
                "split": scenario["split"],
                "controller": "expert" if policy is None else "learned",
                "success": reason == "success",
                "termination": reason,
                "simulation_time_s": elapsed,
                "final_error_m": state["error"],
                "minimum_error_m": float(min(np.min(arrays["error_m"]), state["error"])),
                "base_path_length_m": float(
                    np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum()
                ),
                "arm_joint_travel_rad": float(np.abs(np.diff(arm, axis=0)).sum()),
                "clipped_action_fraction": clipped / len(latency),
                "inference_median_ms": float(np.median(latency)),
                "inference_p95_ms": float(np.percentile(latency, 95)),
                "base_nav_timeout_triggered": bool(policy is None and robot.base_timed_out),
                "control_period_s": robot.control_dt,
                "samples": len(latency),
                "collision_rate": None,
                "collision_note": "Not measured: obstacle/self-collision evaluation not "
                "implemented",
            }
            np.savez_compressed(folder / "trajectory.npz", **arrays)
            (folder / "summary.json").write_text(json.dumps(summary, indent=2))
            (folder / "episode.json").write_text(
                json.dumps(
                    {
                        "scenario": scenario,
                        "settings": config,
                        "source_sha256": provenance(config),
                        "final_ee_m": state["ee"].tolist(),
                    },
                    indent=2,
                )
            )
        return summary
    except Exception as error:
        (folder / "failure.json").write_text(
            json.dumps({"error": repr(error), "scenario": scenario}, indent=2)
        )
        raise
    finally:
        if robot is not None:
            robot.close()
