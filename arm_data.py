"""Physics and data plumbing for the learned arm.

- gravity_torque(q_arm): the 7 joint torques needed just to hold the arm still at angles q_arm (robot model,
  pinocchio). This is physics from the measured joint angles, not an expert internal, and it is the part of the
  expert's torque that the network does NOT learn (torque = gravity_torque(q) + network(inputs)).
- arm_phase_rows(...): picks out the moments where the arm is reaching (the base has parked).
"""
from pathlib import Path
import numpy as np
import pinocchio as pin

URDF = "/home/akhjd/ros2_ws/RoboEnv/models/summit_xl_arm_description/summit_xl_complete_pin.urdf"
_model = pin.buildModelFromUrdf(URDF)   # fixed-base model: q[0:7] are the arm joints
_data = _model.createData()


def gravity_torque(q_arm):
    """Torque (N m) each arm joint needs to hold the arm still at angles q_arm (7,), base level on the floor."""
    q = pin.neutral(_model)
    q[:7] = q_arm
    return pin.computeGeneralizedGravity(_model, _data, q)[:7].copy()

def arm_phase_rows(folder, episode):
    """Indices of the rows of one saved episode where the arm is reaching (from base arrival onwards).

    Obstacle datasets record the arrival time. The course dataset (dataset-v3) does not, so there the reach is
    taken to start at the first moment the expert's wheel commands are all zero after the start.
    """
    z = np.load(Path(folder)/episode["path"]/"trajectory.npz")
    t = z["time_s"]
    arrival = episode.get("base_arrival_time_s")
    if arrival is None:
        stopped = np.all(z["actions"][:, :4] == 0, axis=1) & (t > 0.5)
        if not stopped.any():
            return np.array([], dtype=int)
        arrival = t[np.argmax(stopped)]
    return np.where(t >= arrival)[0]
