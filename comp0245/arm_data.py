"""Physics and data plumbing for the learned arm.

- gravity_torque(q_arm): the 7 joint torques needed just to hold the arm still at angles q_arm
  (robot model, pinocchio). This is physics from the measured joint angles, not an expert internal,
  and it is the part of the expert's torque that the network does NOT learn (torque =
  gravity_torque(q) + network(inputs)).
- arm_phase_rows(...): picks out the moments where the arm is reaching (the base has parked).
"""

import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from comp0245.paths import SETTINGS_FILE

URDF = str(
    Path(json.loads(SETTINGS_FILE.read_text())["roboenv_root"])
    / "models/summit_xl_arm_description/summit_xl_complete_pin.urdf"
)
_model = pin.buildModelFromUrdf(URDF)  # fixed-base model: q[0:7] are the arm joints
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
    # TODO 1: pin.computeFrameJacobian(_model, _data, q, EE_FRAME, pin.LOCAL_WORLD_ALIGNED),
    #         keep only the position rows (first 3) and the arm columns (first 7)
    J = ...  # noqa: F841 (placeholder until TODO 1 is written)
    # TODO 2: damped pseudoinverse times the error:  J^T (J J^T + DAMPING^2 I)^-1 e
    #         (np.linalg.solve(A, b) computes A^-1 b; np.eye(3) is the 3x3 identity)
    return ...


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
