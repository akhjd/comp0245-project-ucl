"""Closed-loop test of a learned lidar base policy on obstacle scenarios: the model drives the
wheels alone.

The arm is NOT learned here: a fixed PD holds it at its starting pose (same as eval_base.py), so
the hand never reaches and the "success" flag in summary.json is false by design. What is scored is
the base: did it park at the reach-zone distance facing the target, without touching anything, and
could the arm reach from there?

Example:
    python -m comp0245.evaluation.eval_obstacles --model runs/obst-base-torch-v1/model.pt --out \\
        runs/obst-base-torch-v1-validation
    python -m comp0245.apps.view_runs runs/obst-base-torch-v1-validation --expert runs/obst-v2b
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from comp0245.models import base_inputs
from comp0245.paths import SCENARIO_DIR, SETTINGS_FILE, scenario_path
from comp0245.pipeline import read_json, write_json
from comp0245.sim.obstacle_env import SteerAround, rollout, standoff, wrap
from comp0245.sim.reach_zone import ARM_REACH, SHOULDER_X, SHOULDER_Z
from comp0245.training.train_base_torch import INPUT_NAMES

KP, KD = 100.0, 30.0  # arm hold, same stiffness as the expert's arm controller
SETTLE_WINDOW_S = (
    2.0  # the parking checks must hold for this whole final stretch, not at one moment
)
MAX_SPEED, MAX_YAW_RATE = 0.02, 0.05  # m/s, rad/s: "stopped"


def load_model(path):
    """Rebuild the network from model.pt (same layers as comp0245/training/train_base_torch.py)
    and its scaling numbers. Loads on the CPU even if it was trained on a GPU, and refuses a
    model whose inputs differ from ours."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    expected = [INPUT_NAMES[i] for i in saved.get("columns") or range(len(INPUT_NAMES))]
    if saved["input_names"] != expected:
        raise ValueError(
            f"{path}: model inputs {saved['input_names']} do not match the evaluator's {expected}"
        )
    h = saved["hidden"]
    n_in = len(saved["input_names"])
    model = nn.Sequential(
        nn.Linear(n_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 4)
    )
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved


class LearnedBase:
    """policy.act(state) -> 11 commands: 4 wheel speeds from the network, 7 arm torques holding
    the start pose. Make a new one for every episode (it remembers the arm's starting pose)."""

    def __init__(self, model, saved):
        self.model = model
        self.x_mean, self.x_std = saved["x_mean"], saved["x_std"]
        self.y_mean, self.y_std = saved["y_mean"], saved["y_std"]
        self.columns = saved.get(
            "columns"
        )  # optional: which of the 45 inputs this model was trained on
        self.q_hold = None

    def act(self, state):
        q, qd = state["q"][4:11], state["qd"][4:11]
        if self.q_hold is None:
            self.q_hold = q.copy()
        # Same 45 inputs, in the same order, as training: 9 base features then 36 lidar ranges
        x = np.hstack([base_inputs(state["x"]), state["lidar"]]).astype(np.float32)
        if self.columns is not None:
            x = x[self.columns]
        with torch.no_grad():
            out = self.model(torch.from_numpy((x - self.x_mean) / self.x_std)[None])[0].numpy()
        wheels = out * self.y_std + self.y_mean  # undo the output scaling -> rad/s
        arm = KP * (self.q_hold - q) - KD * qd
        return np.concatenate([wheels, arm])


class ExpertHold:
    """Like-for-like reference for the learned bases: the expert's wheels (SteerAround,
    sensor-only) with the same fixed PD arm hold and episode length as LearnedBase. The
    full-task expert (policy=None) also reaches with a gravity-compensated arm and ends the
    episode early, which changes what the lidar sees and how long the robot is exposed, so it is
    not a fair base comparison. Make a new one for every episode."""

    def __init__(self, target):
        self.target = np.asarray(target, dtype=float)
        self.base = SteerAround()
        self.arrived = False
        self.q_hold = None

    def act(self, state):
        q, qd = state["q"][4:11], state["qd"][4:11]
        if self.q_hold is None:
            self.q_hold = q.copy()
        wheels = np.zeros(4)
        if not self.arrived:  # like the full-task expert: once parked, the wheels stay stopped
            wheels, self.arrived = self.base.wheels(
                state["base"][:2], state["yaw"], state["x"][22:24], state["lidar"], self.target
            )
            if self.arrived:
                wheels = np.zeros(4)
        arm = KP * (self.q_hold - q) - KD * qd
        return np.concatenate([wheels, arm])


def score(folder, scenario, summary):
    """Base parking quality from the saved trajectory. Strict/loose need the parking to HOLD:
    every saved row of the last SETTLE_WINDOW_S seconds (and the final state) must be within the
    distance and facing limits, with the base stopped (linear speed and yaw rate)."""
    z = np.load(folder / "trajectory.npz")
    obs, t = z["observations"], z["time_s"]
    target = np.array(scenario["target"])
    base, yaw = obs[-1, 0:2], np.arctan2(obs[-1, 3], obs[-1, 4])
    end = t >= t[-1] - SETTLE_WINDOW_S
    to_target = target[:2] - obs[end, 0:2]
    window_distance = np.abs(np.linalg.norm(to_target, axis=1) - standoff(target[2]))
    window_facing = np.abs(
        wrap(np.arctan2(to_target[:, 1], to_target[:, 0]) - np.arctan2(obs[end, 3], obs[end, 4]))
    )
    distance_error = summary["final_standoff_error_m"]
    facing_error = summary["final_facing_error_rad"]
    worst_distance = max(float(window_distance.max()), distance_error)
    worst_facing = max(float(window_facing.max()), facing_error)
    speed = float(np.hypot(obs[-1, 22], obs[-1, 23]))
    max_speed = float(np.hypot(obs[end, 22], obs[end, 23]).max())
    max_yaw_rate = float(np.abs(obs[end, 27]).max())
    still = max_speed < MAX_SPEED and max_yaw_rate < MAX_YAW_RATE
    # Could the arm reach the target from where the base stopped? (shoulder-centred reach sphere)
    shoulder = base + SHOULDER_X * np.array([np.cos(yaw), np.sin(yaw)])
    reach_needed = float(np.hypot(np.linalg.norm(target[:2] - shoulder), target[2] - SHOULDER_Z))
    # When did it come to rest for good? Last time the base was still moving faster than 2 cm/s
    v = np.hypot(obs[:, 22], obs[:, 23])
    moving = np.where(v > 0.02)[0]
    settled_at = float(t[moving[-1]]) if len(moving) else 0.0
    contact = summary["contact_steps"] > 0
    return {
        "id": scenario["id"],
        "distance_error_m": distance_error,
        "facing_error_rad": facing_error,
        "final_speed_m_s": speed,
        "settled_at_s": settled_at,
        "window_s": SETTLE_WINDOW_S,
        "worst_distance_error_m": worst_distance,
        "worst_facing_error_rad": worst_facing,
        "max_speed_m_s": max_speed,
        "max_yaw_rate_rad_s": max_yaw_rate,
        "reach_needed_m": reach_needed,
        "reachable": reach_needed <= ARM_REACH,
        "contact_steps": summary["contact_steps"],
        "min_clearance_m": summary["min_clearance_m"],
        "base_path_length_m": summary["base_path_length_m"],
        "strict": (not contact) and worst_distance <= 0.05 and worst_facing <= 0.05 and still,
        "loose": (not contact) and worst_distance <= 0.15 and worst_facing <= 0.2 and still,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scenarios", default=str(SCENARIO_DIR / "scenarios_obstacles_v2.json"))
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--duration",
        type=float,
        default=40.0,
        help="seconds per episode (expert parks in 15-24 s)",
    )
    args = parser.parse_args()
    if args.split == "test":
        raise SystemExit(
            "The test split is reserved for the final evaluation; pass --split test only then."
        )
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Choose a new output folder: {out}")
    torch.set_num_threads(1)  # one row per step: threads only add overhead
    model, saved = load_model(args.model)
    config = read_json(SETTINGS_FILE)
    config["duration_s"] = args.duration
    scenarios = [
        s
        for s in json.loads(scenario_path(args.scenarios).read_text())
        if s["split"] == args.split
    ]
    out.mkdir(parents=True)
    results = []
    for s in scenarios:
        start = time.time()
        summary = rollout(config, s, out / s["id"], LearnedBase(model, saved))
        r = score(out / s["id"], s, summary)
        r["inference_ms_median"] = float(
            np.median(np.load(out / s["id"] / "trajectory.npz")["inference_ms"])
        )
        results.append(r)
        print(
            f"{r['id']}: parking error {r['distance_error_m']:.3f} m, facing "
            f"{r['facing_error_rad']:.3f} rad, "
            f"contacts {r['contact_steps']}, clearance {r['min_clearance_m']:.2f} m, "
            f"settled {r['settled_at_s']:.1f} s, reachable {r['reachable']}, "
            f"strict {r['strict']}, loose {r['loose']}  ({time.time() - start:.0f} s)",
            flush=True,
        )
    n = len(results)
    totals = {
        "episodes": n,
        "strict": sum(r["strict"] for r in results),
        "loose": sum(r["loose"] for r in results),
        "no_contact": sum(r["contact_steps"] == 0 for r in results),
        "reachable": sum(r["reachable"] for r in results),
    }
    print(
        f"strict {totals['strict']}/{n}, loose {totals['loose']}/{n}, no contact "
        f"{totals['no_contact']}/{n}, "
        f"arm could reach {totals['reachable']}/{n}"
    )
    write_json(
        out / "evaluation.json",
        {
            "model": str(args.model),
            "scenarios": args.scenarios,
            "split": args.split,
            "duration_s": args.duration,
            "arm": "fixed PD hold (not learned)",
            "totals": totals,
            "episodes": results,
        },
    )


if __name__ == "__main__":
    main()
