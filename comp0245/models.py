"""Shared state encoding and interchangeable supervised policies. No simulator imports."""

from collections import deque

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = (
    ["base_x_m", "base_y_m", "base_z_m", "sin_yaw", "cos_yaw"]
    + ["target_x_m", "target_y_m", "target_z_m"]
    + [f"arm_q_{j}_rad" for j in range(7)]
    + [f"arm_qd_{j}_rad_s" for j in range(7)]
    + [f"base_v_{a}_m_s" for a in "xyz"]
    + [f"base_w_{a}_rad_s" for a in "xyz"]
)
ACTION_NAMES = [
    f"wheel_{w}_rad_s" for w in ["front_right", "front_left", "back_left", "back_right"]
] + [f"arm_tau_{j}_nm" for j in range(7)]
SCHEMA_VERSION = 1


def observation(base, yaw, target, q, qd, linear_velocity, angular_velocity):
    """Build the 28-number observation (order of FEATURE_NAMES); raises if any value is not finite.

    base (3,) m, yaw rad, target (3,) m in the world frame, arm q (7,) rad and qd (7,) rad/s, base
    linear (3,) m/s and angular (3,) rad/s velocity in the world frame.
    """
    x = np.concatenate(
        [base, [np.sin(yaw), np.cos(yaw)], target, q, qd, linear_velocity, angular_velocity]
    ).astype(np.float64)
    if x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all():
        raise ValueError("Invalid observation shape or nonfinite robot state")
    return x


def target_in_base_frame(base, yaw, target):
    """Target position seen from the robot: (forward, left, up) in metres."""
    dx = target[0] - base[0]
    dy = target[1] - base[1]
    forward = np.cos(yaw) * dx + np.sin(yaw) * dy
    left = np.cos(yaw) * dy - np.sin(yaw) * dx  # -sin(yaw)*dx + cos(yaw)*dy
    up = target[2] - base[2]
    return np.array([forward, left, up])


def base_features(base, yaw, target):
    """What the base needs: where the target is, how far, and how much to turn."""
    forward, left, up = target_in_base_frame(base, yaw, target)
    distance = np.hypot(forward, left)  # horizontal distance, m
    heading_error = np.arctan2(left, forward)  # 0 = target dead ahead, + = turn left
    return np.array([forward, left, distance, np.sin(heading_error), np.cos(heading_error)])


BASE_INPUT_NAMES = [
    "forward_m",
    "left_m",
    "distance_m",
    "sin_heading_err",
    "cos_heading_err",
    "up_m",
    "v_forward_m_s",
    "v_left_m_s",
    "yaw_rate_rad_s",
]


def base_inputs(obs):
    """One 28-number observation -> the 9 numbers the base policy sees."""
    base = obs[0:3]
    yaw = np.arctan2(obs[3], obs[4])
    target = obs[5:8]
    vx, vy = obs[22], obs[23]
    v_forward = (
        np.cos(yaw) * vx + np.sin(yaw) * vy
    )  # same rotation as `forward`, but applied to (vx, vy)
    v_left = np.cos(yaw) * vy - np.sin(yaw) * vx  # same rotation as `left`
    up = obs[7] - obs[2]
    return np.concatenate([base_features(base, yaw, target), [up, v_forward, v_left, obs[27]]])


SHOULDER = np.array(
    [0.204, 0.0, 0.496]
)  # shoulder position in the base frame (m), same as comp0245/sim/reach_zone.py

ARM_INPUT_NAMES = (
    [f"sin_q{j}" for j in (0, 2, 4, 6)]
    + [f"cos_q{j}" for j in (0, 2, 4, 6)]
    + [f"q{j}_rad" for j in (1, 3, 5)]
    + [f"qd{j}_rad_s" for j in range(7)]
    + ["hand_to_target_forward_m", "hand_to_target_left_m", "hand_to_target_up_m"]
    + ["shoulder_to_target_forward_m", "shoulder_to_target_left_m", "shoulder_to_target_up_m"]
)


def arm_inputs(obs, ee):
    """One 28-number observation + hand position (3,) -> the 24 numbers the arm policy sees."""
    base = obs[0:3]
    yaw = np.arctan2(obs[3], obs[4])
    target = obs[5:8]
    q, qd = obs[8:15], obs[15:22]
    spinning = q[[0, 2, 4, 6]]
    limited = q[[1, 3, 5]]
    hand_to_target = target_in_base_frame(ee, yaw, target)  # TODO 1
    shoulder_to_target = target_in_base_frame(base, yaw, target) - SHOULDER  # TODO 2
    return np.concatenate(
        [np.sin(spinning), np.cos(spinning), limited, qd, hand_to_target, shoulder_to_target]
    )


class History:
    """Oldest to newest; pad a new episode with its first observation, never the previous
    episode."""

    def __init__(self, length):
        """Keep the last `length` observations."""
        if length < 1:
            raise ValueError("history must be positive")
        self.length, self.values = length, deque(maxlen=length)

    def append(self, x):
        """Add one observation (n,) and return the stacked history (length * n,), oldest first."""
        if not self.values:
            self.values.extend([x.copy() for _ in range(self.length - 1)])
        self.values.append(x.copy())
        return np.concatenate(self.values)


def stack_episode(x, length):
    """History rows for a whole episode: (T, n) -> (T, length * n), causal within the episode."""
    history = History(length)
    return np.stack([history.append(row) for row in x])


def clip_action(action, config):
    """Clip an (11,) command to the wheel (rad/s) and arm torque (N m) limits in the settings."""
    action = np.asarray(action, dtype=float)
    if action.shape != (11,) or not np.isfinite(action).all():
        raise ValueError("Policy must return 11 finite commands")
    limits = np.array([config["wheel_limit_rad_s"]] * 4 + [config["arm_limit_nm"]] * 7)
    return np.clip(action, -limits, limits)


def make_model(name, config):
    """Register another estimator here; it must implement fit(X, Y) and predict(X)."""
    opts = config["models"][name].copy()
    if name == "ridge":
        estimator = Ridge(**opts)
    elif name == "mlp":
        opts["hidden_layer_sizes"] = tuple(opts["hidden_layer_sizes"])
        estimator = MLPRegressor(random_state=config["seed"], **opts)
    elif name == "forest":
        estimator = RandomForestRegressor(random_state=config["seed"], **opts)
    else:
        raise ValueError(f"Unknown model: {name}")
    # Both scalers are fitted only by fit() on the training episodes.
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), estimator), transformer=StandardScaler()
    )


def regression_metrics(y, prediction):
    """Offline RMSE: wheels (rad/s), arm (N m) and per output. y, prediction: (rows, 11)."""
    mse = np.mean((np.asarray(y) - prediction) ** 2, axis=0)
    return {
        "wheel_rmse_rad_s": float(np.sqrt(np.mean(mse[:4]))),
        "arm_rmse_nm": float(np.sqrt(np.mean(mse[4:]))),
        "per_action_rmse": dict(zip(ACTION_NAMES, np.sqrt(mse).tolist(), strict=False)),
    }
