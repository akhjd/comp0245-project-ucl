"""Closed-loop test of a learned lidar base policy on obstacle scenarios: the model drives the
wheels alone.

The arm is NOT learned here: a fixed PD holds it at its starting pose (same as eval_base.py), so
the hand never reaches and the "success" flag in summary.json is false by design. What is scored is
the base. Reported since 3 Nov: "parked" = it stopped where the arm can reach the target (measured
reach area), looking towards it, still for the last 2 s, without touching anything. Strict/loose
(at the expert's reach-zone distance, 5 / 15 cm) are kept for the record only: off the expert's
route the best stopping place differs, so matching it is not required.

Example:
    python -m comp0245.evaluation.eval_obstacles --model runs/obst-base-torch-v1/model.pt --out \\
        runs/obst-base-torch-v1-validation
    python -m comp0245.apps.view_runs runs/obst-base-torch-v1-validation --expert runs/obst-v2b
"""

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import torch

from comp0245.models import base_inputs
from comp0245.paths import SCENARIO_DIR, SETTINGS_FILE, scenario_path
from comp0245.pipeline import read_json, write_json
from comp0245.sim.obstacle_env import SteerAround, rollout, standoff, wrap
from comp0245.sim.reach_zone import ARM_REACH, SHOULDER_X, SHOULDER_Z
from comp0245.training.train_base_torch import SCAN_GAP_S, build_net, input_names, target_ray

KP, KD = 100.0, 30.0  # arm hold, same stiffness as the expert's arm controller
STICK = 2.0  # mixture model: switch candidate only when another is this many times as likely
# Arm pose while the base drives with the arm folded (scenario "arm_q0"): every arm part inside the
# wheel-box
# footprint (x +-0.342, y +-0.313 m), 9 cm from the robot's own body (runs/arm-tuck/search.py, 4
# Nov)
ARM_FOLD = [3.14159, 2.618, 3.14159, 0.524, 0.0, 1.309, 0.0]
# Modes model: hysteresis on the switch's evade probability. Enter evade above SWITCH_ON, go back
# to the target head
# only below SWITCH_OFF; in between the current mode stays, so one noisy step does not flip the
# head
SWITCH_ON, SWITCH_OFF = 0.6, 0.4
# Hold output (comp0245/training/train_hold.py): above this probability of "the expert commands
# exactly 0 here" the wheels get exactly 0
HOLD_ON = 0.5
# Latched hold (model saved with "hold_latch"): once the probability has stayed above HOLD_ON for
# HOLD_LATCH_S, keep
# holding until it drops below HOLD_OFF, so a few low moments do not move a parked base (4 Nov,
# rmx_023)
HOLD_LATCH_S, HOLD_OFF = 0.5, 0.2
SETTLE_WINDOW_S = (
    2.0  # the parking checks must hold for this whole final stretch, not at one moment
)
MAX_SPEED, MAX_YAW_RATE = 0.02, 0.05  # m/s, rad/s: "stopped"
# "parked" (the reported base score since 3 Nov; strict/loose are still computed for the record):
# the base stops
# where the expert arm can reach the target, looking towards it. Distances are horizontal, base
# centre to target,
# measured with the expert arm from a still base (comp0245/evaluation/measure_reach.py: runs/reach-
# area, runs/reach-area-fine).
# Measured edge with the target 30 deg to the side (the facing limit, the worst case): the arm
# still reached at 1.02 m
# for targets 0.36 and 0.58 m high and at 0.98 m for 0.80 m, about 83% of the straight arm at every
# height; straight
# ahead it reached 2-4 cm further. Accepted: 83%, so 0.98-1.02 m for the room targets (0.36-0.80 m
# high). Not less:
# the expert aims at 75% (reach_zone.COMFORT), parks up to 5 cm beyond (about 80%) and its arm
# reaches from there,
# while the base is nudged a few mm back as the arm reaches out.
REACH_MIN_M = 0.3  # closest distance tried; the arm reached at every height
REACH_SHARE = 0.83  # share of the straight arm's horizontal reach at the target's height
FACING_LIMIT = float(np.radians(30))  # target within +-30 deg of straight ahead
# 4 Nov (Akhilesh: "measured reach"): the same measurements straight ahead reached 1.04 / 1.06 /
# 1.00 m for targets
# 0.36 / 0.58 / 0.80 m high, 85.7 / 87.2 / 84.96% of the straight arm. Accepted: 84.9% facing the
# target exactly, falling
# linearly to 83% at the facing limit, so the limit never passes a measured edge. Every model is
# rescored with it;
# "parked_worst_case_reach" keeps the 83%-everywhere rule for the record.
REACH_SHARE_AHEAD = 0.849


def reach_limit(target_z, facing=FACING_LIMIT):
    """Farthest accepted parking distance (m, horizontal, base centre to target) for a target at
    this height, with the target `facing` rad off straight ahead (default: the facing limit, the
    worst case). Works on arrays of facing."""
    arm = np.sqrt(max(ARM_REACH**2 - (target_z - SHOULDER_Z) ** 2, 0.0))
    off = np.minimum(np.abs(facing) / FACING_LIMIT, 1.0)
    return SHOULDER_X + (REACH_SHARE_AHEAD + (REACH_SHARE - REACH_SHARE_AHEAD) * off) * arm


def load_model(path):
    """Rebuild the network from model.pt (same layers as comp0245/training/train_base_torch.py)
    and its scaling numbers. Loads on the CPU even if it was trained on a GPU, and refuses a
    model whose inputs differ from ours."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    names = input_names(
        saved.get("scans", 1), saved.get("rays", 36), saved.get("target_ray", False)
    )  # before "rays": 36
    expected = [names[i] for i in saved.get("columns") or range(len(names))]
    if saved["input_names"] != expected:
        raise ValueError(
            f"{path}: model inputs {saved['input_names']} do not match the evaluator's {expected}"
        )
    h = saved["hidden"]
    n_in = len(saved["input_names"])
    model = build_net(
        n_in,
        h,
        saved.get("scans", 1),
        saved.get("encoder", "mlp"),
        saved.get("threat", False),
        saved.get("mixture", 0),
        saved.get("rays", 36),
        saved.get("modes", False),
        saved.get("menu", False),
        saved.get("hold", False),
        saved.get("dropout", 0.0),
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
        )  # optional: which of the inputs this model was trained on
        self.scans, self.gap = (
            saved.get("scans", 1),
            int(round(SCAN_GAP_S / 0.001)),
        )  # act() runs every 1 ms
        self.frames = collections.deque(maxlen=(self.scans - 1) * self.gap + 1)
        self.q_hold = None
        self.choice = None  # mixture model: the candidate being followed
        self.rays = saved.get("rays", 36)
        self.target_ray = saved.get(
            "target_ray", False
        )  # "target blocked?" inputs from the current scan (5 Nov)
        self.evading = False  # modes model: which head drives (starts on the target head)
        self.mode_log, self.switch_log = (
            [],
            [],
        )  # modes model: per step, evade head used and switch probability
        self.hold_log = []  # hold output: per step, probability of "hold still"
        self.hold_latch = saved.get("hold_latch", False)
        self.hold_run, self.latched = (
            0,
            False,
        )  # latched hold: steps in a row above HOLD_ON, and whether latched

    def act(self, state):
        q, qd = state["q"][4:11], state["qd"][4:11]
        if self.q_hold is None:
            self.q_hold = q.copy()
        # Same inputs, in the same order, as training: 9 base features then the lidar ranges of
        # each scan
        if len(state["lidar"]) != self.rays:
            raise ValueError(
                f"the model was trained on {self.rays} lidar rays but the environment gives "
                f"{len(state['lidar'])}: set config lidar_rays (--lidar-rays) to {self.rays}"
            )
        self.frames.append(np.array(state["lidar"]))
        newest = len(self.frames) - 1
        lidar = [self.frames[max(newest - k * self.gap, 0)] for k in range(self.scans)]
        base = base_inputs(state["x"])
        if self.target_ray:
            base = np.concatenate([base, target_ray(base, lidar[0])[0]])
        x = np.hstack([base, *lidar]).astype(np.float32)
        if self.columns is not None:
            x = x[self.columns]
        with torch.no_grad():
            xt = torch.from_numpy((x - self.x_mean) / self.x_std)[None]
            if getattr(self.model, "mixture", 0):
                out = self.pick(
                    *(a[0].numpy() for a in self.model.candidates(self.model.features(xt))[:2])
                )
            elif getattr(self.model, "modes", False):
                logit, both = self.model.heads(self.model.features(xt))
                out = both[0, int(self.switch(float(torch.sigmoid(logit[0]))))].numpy()
            elif getattr(self.model, "hold", None) is not None:
                h = self.model.features(xt)
                out = self.model.wheels(h)[0].numpy()
                self.hold_log.append(float(torch.sigmoid(self.model.hold(h))[0, 0]))
            else:
                out = self.model(xt)[0].numpy()
        wheels = out * self.y_std + self.y_mean  # undo the output scaling -> rad/s
        if self.hold_log and self.holding(self.hold_log[-1]):
            wheels = np.zeros(4)  # the expert's parked / waiting command: exactly still
        arm = KP * (self.q_hold - q) - KD * qd
        return np.concatenate([wheels, arm])

    def holding(self, p_hold):
        """Hold output: drive exactly 0 now? Above HOLD_ON; latched models also keep holding
        once latched."""
        self.hold_run = self.hold_run + 1 if p_hold > HOLD_ON else 0
        if self.hold_latch:
            if self.hold_run >= HOLD_LATCH_S / 0.001:  # act() runs every 1 ms
                self.latched = True
            elif p_hold < HOLD_OFF:
                self.latched = False
        return p_hold > HOLD_ON or self.latched

    def switch(self, p_evade):
        """Modes model: evade head from the switch probability with hysteresis (SWITCH_ON /
        SWITCH_OFF); logged."""
        if self.evading and p_evade < SWITCH_OFF:
            self.evading = False
        elif not self.evading and p_evade > SWITCH_ON:
            self.evading = True
        self.mode_log.append(self.evading)
        self.switch_log.append(p_evade)
        return self.evading

    def save_modes(self, folder):
        """Modes model: save the per-step mode (True = evade head) and switch probability as
        folder/modes.npz. Hold output: the per-step hold probability as folder/hold.npz."""
        if self.hold_log:
            np.savez_compressed(
                Path(folder) / "hold.npz", hold_p=np.array(self.hold_log, np.float32)
            )
        if self.mode_log:
            np.savez_compressed(
                Path(folder) / "modes.npz",
                evade=np.array(self.mode_log, bool),
                switch_p=np.array(self.switch_log, np.float32),
            )

    def pick(self, log_p, mean):
        """Follow the most likely candidate, but keep the current one until another is at least
        STICK times as likely, so the base does not flip between going left and going right from
        one millisecond to the next."""
        best = int(log_p.argmax())
        if self.choice is None or log_p[best] - log_p[self.choice] > np.log(STICK):
            self.choice = best
        return mean[self.choice]


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
    """Base parking quality from the saved trajectory. Parked/strict/loose need the parking to
    HOLD: every saved row of the last SETTLE_WINDOW_S seconds (and the final state) must be
    within the distance and facing limits, with the base stopped (linear speed and yaw rate),
    and there must be no contact in the whole run."""
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
    # Parked: inside the reach area and facing the target on every row of the window, and at the
    # end
    window_range = np.linalg.norm(to_target, axis=1)
    in_area = bool(
        np.all(
            (window_range >= REACH_MIN_M) & (window_range <= reach_limit(target[2], window_facing))
        )
    )
    in_area_worst = bool(
        np.all((window_range >= REACH_MIN_M) & (window_range <= reach_limit(target[2])))
    )
    facing_ok = max(float(window_facing.max()), facing_error) <= FACING_LIMIT
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
        "range_m": float(window_range[-1]),
        "in_reach_area": in_area,
        "parked": bool((not contact) and in_area and facing_ok and still),
        "parked_worst_case_reach": bool((not contact) and in_area_worst and facing_ok and still),
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
    parser.add_argument(
        "--lidar-rays", type=int, help="lidar rays in the environment (default: the model's own)"
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
    config["lidar_rays"] = args.lidar_rays or saved.get("rays", 36)
    config["lidar_range"] = saved.get("lidar_range", 4.0)
    scenarios = [
        s
        for s in json.loads(scenario_path(args.scenarios).read_text())
        if s["split"] == args.split
    ]
    out.mkdir(parents=True)
    results = []
    for s in scenarios:
        start = time.time()
        policy = LearnedBase(model, saved)
        summary = rollout(config, s, out / s["id"], policy)
        policy.save_modes(out / s["id"])
        r = score(out / s["id"], s, summary)
        r["inference_ms_median"] = float(
            np.median(np.load(out / s["id"] / "trajectory.npz")["inference_ms"])
        )
        if (
            policy.mode_log
        ):  # modes model: share of steps on the evade head, and how often the head changed
            r["evade_head_share"] = float(np.mean(policy.mode_log))
            r["head_changes"] = int(np.count_nonzero(np.diff(policy.mode_log)))
        results.append(r)
        print(
            f"{r['id']}: parking error {r['distance_error_m']:.3f} m, facing "
            f"{r['facing_error_rad']:.3f} rad, "
            f"contacts {r['contact_steps']}, clearance {r['min_clearance_m']:.2f} m, "
            f"settled {r['settled_at_s']:.1f} s, reachable {r['reachable']}, "
            f"parked {r['parked']} (strict {r['strict']}, loose {r['loose']})  "
            f"({time.time() - start:.0f} s)",
            flush=True,
        )
    n = len(results)
    totals = {
        "episodes": n,
        "parked": sum(r["parked"] for r in results),
        "strict": sum(r["strict"] for r in results),
        "loose": sum(r["loose"] for r in results),
        "no_contact": sum(r["contact_steps"] == 0 for r in results),
        "reachable": sum(r["reachable"] for r in results),
    }
    print(
        f"parked {totals['parked']}/{n} (strict {totals['strict']}, loose {totals['loose']}), "
        f"no contact {totals['no_contact']}/{n}, arm could reach {totals['reachable']}/{n}"
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
