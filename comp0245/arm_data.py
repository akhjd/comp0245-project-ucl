"""Physics and data plumbing for the learned arm.

- gravity_torque(q_arm): the 7 joint torques needed just to hold the arm still at angles q_arm
  (robot model, pinocchio). This is physics from the measured joint angles, not an expert internal,
  and it is the part of the expert's torque that the network does NOT learn (torque =
  gravity_torque(q) + network(inputs)).
- ik_direction(q_arm, hand_to_target): 7 joint speeds that move the hand towards the target (model
  B's extra inputs).
- arm_phase_rows(...): picks out the moments where the arm is reaching (the base has parked).
"""

import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from comp0245.models import ARM_INPUT_NAMES, arm_inputs
from comp0245.paths import SETTINGS_FILE

ROBOENV_ROOT = Path(json.loads(SETTINGS_FILE.read_text())["roboenv_root"])
URDF = ROBOENV_ROOT / "models/summit_xl_arm_description/summit_xl_complete_pin.urdf"
_model = pin.buildModelFromUrdf(str(URDF))  # fixed-base model: q[0:7] are the arm joints
_data = _model.createData()


def gravity_torque(q_arm):
    """Torque (N m) each arm joint needs to hold the arm still at angles q_arm (7,), base level
    on the floor."""
    q = pin.neutral(_model)
    q[:7] = q_arm
    return pin.computeGeneralizedGravity(_model, _data, q)[:7].copy()


EE_FRAME = _model.getFrameId("robot_j2s7s300_end_effector")  # the hand
DAMPING = 0.05  # m: keeps the speeds sane near singular poses (the expert uses 0.01-0.10)


def ik_direction(q_arm, hand_to_target):
    """7 joint speeds that move the hand along hand_to_target (robot's directions), least
    effort: J+ e."""
    q = pin.neutral(_model)
    q[:7] = q_arm
    J = pin.computeFrameJacobian(_model, _data, q, EE_FRAME, pin.LOCAL_WORLD_ALIGNED)[
        :3, :7
    ]  # hand position rows, arm columns
    return J.T @ np.linalg.solve(
        J @ J.T + DAMPING**2 * np.eye(3), hand_to_target
    )  # damped pseudoinverse times the error


# Model B: model A's 24 inputs + the 7 joint speeds that would move the hand straight at the target
ARM_B_INPUT_NAMES = ARM_INPUT_NAMES + [f"ik_qd{j}_rad_s" for j in range(7)]


def arm_inputs_b(obs, ee):
    """One 28-number observation + hand position (3,) -> the 31 numbers model B sees."""
    x = arm_inputs(obs, ee)
    return np.concatenate(
        [x, ik_direction(obs[8:15], x[-6:-3])]
    )  # x[-6:-3] = hand to target, robot's directions


def arm_input_fn(names):
    """The input function that matches a saved model's input names (model A or model B)."""
    if list(names) == ARM_INPUT_NAMES:
        return arm_inputs
    if list(names) == ARM_B_INPUT_NAMES:
        return arm_inputs_b
    raise ValueError(f"arm inputs {list(names)} match neither model A nor model B")


def arm_phase_rows(folder, episode):
    """Indices of the rows of one saved episode where the arm is reaching (from base arrival
    onwards).

    Obstacle datasets record the arrival time. The course dataset (dataset-v3) does not, so
    there the reach is taken to start at the first moment the expert's wheel commands are all
    zero after the start.
    """
    z = np.load(Path(folder) / episode["path"] / "trajectory.npz")
    t = z["time_s"]
    arrival = episode.get("base_arrival_time_s")
    if arrival is None:
        stopped = np.all(z["actions"][:, :4] == 0, axis=1) & (t > 0.5)
        if not stopped.any():
            return np.array([], dtype=int)
        arrival = t[np.argmax(stopped)]
    return np.where(t >= arrival)[0]
