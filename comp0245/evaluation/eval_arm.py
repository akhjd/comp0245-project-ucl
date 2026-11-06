"""Closed-loop test of a learned arm: does the hand actually reach the target when the network
controls the arm?

The expert's wheels drive and park (same logic as eval_obstacles.ExpertHold), so a miss is the
arm's fault and not bad parking. While driving, the arm holds its start pose like the expert does
(PD + gravity). Once the base has parked, the arm torque is
    gravity_torque(q) + network(arm_inputs)          (comp0245/training/train_arm.py)
The episode ends with the course's success rule (hand within 1 cm for 0.25 s), or after --duration.
--base MODEL.pt drives with a learned base instead (the fully learned robot); it counts as parked
once the base has been still for 1 s.

Scenarios: the arm's held-out episodes (every 10th successful episode of each dataset, never
trained on) and validationc_030-059 (fresh, normal heights). The test split is never used.

Example:
    python -m comp0245.evaluation.eval_arm --name arm-a --arm runs/arm-a/model.pt
"""

import argparse
import json
import multiprocessing as mp
import os

import numpy as np

from comp0245.paths import ROOT, SETTINGS_FILE, scenario_path

ARM_DATA = {
    "obst-v4-height": "scenarios_obstacles_v4_height.json",
    "obst-v4-armstart": "scenarios_obstacles_v4_armstart.json",
    "obst-v4-both": "scenarios_obstacles_v4_both.json",
}
KP, KD = 100.0, 30.0  # the expert's arm stiffness
STILL_S = (
    1.0  # learned base: parked once still (speed < 0.02 m/s, yaw rate < 0.05 rad/s) this long
)


def load_arm(path):
    """Rebuild the arm network from model.pt (same layers as comp0245/training/train_arm.py)."""
    import torch
    from torch import nn

    from comp0245.models import ARM_INPUT_NAMES

    saved = torch.load(path, map_location="cpu", weights_only=False)
    if list(saved["input_names"]) != ARM_INPUT_NAMES:
        raise ValueError(
            f"{path}: arm inputs {saved['input_names']} do not match models.ARM_INPUT_NAMES"
        )
    h, n = saved["hidden"], len(ARM_INPUT_NAMES)
    model = nn.Sequential(nn.Linear(n, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 7))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved


class LearnedArm:
    """policy.act(state) -> 11 commands. Base: expert wheels (or a learned base); arm: hold,
    then learned reach. Make a new one for every episode."""

    def __init__(self, arm, arm_saved, target, base=None):
        from comp0245.evaluation.eval_obstacles import ExpertHold

        self.arm, self.s = arm, arm_saved
        self.base = base or ExpertHold(target)
        self.q_hold = None
        self.reach_from = None  # time (s) the arm started reaching
        self.t, self.still = 0.0, 0.0

    def parked(self, state):
        if hasattr(self.base, "arrived"):
            return self.base.arrived
        moving = np.hypot(*state["x"][22:24]) > 0.02 or abs(state["x"][27]) > 0.05
        self.still = 0.0 if moving or self.t < 2.0 else self.still + 0.001
        return self.still >= STILL_S

    def act(self, state):
        import torch

        from comp0245.arm_data import gravity_torque
        from comp0245.models import arm_inputs

        q, qd = state["q"][4:11], state["qd"][4:11]
        if self.q_hold is None:
            self.q_hold = q.copy()
        wheels = self.base.act(state)[:4]
        if self.reach_from is None and self.parked(state):
            self.reach_from = self.t
        if self.reach_from is None:  # driving: hold the start pose like the expert (PD + gravity)
            arm = KP * (self.q_hold - q) - KD * qd + gravity_torque(q)
        else:
            wheels = np.zeros(4)
            x = (arm_inputs(state["x"], state["ee"]) - self.s["x_mean"]) / self.s["x_std"]
            with torch.no_grad():
                out = self.arm(torch.from_numpy(x.astype(np.float32))[None])[0].numpy()
            arm = gravity_torque(q) + out * self.s["y_std"] + self.s["y_mean"]
        self.t += 0.001
        return np.concatenate([wheels, arm])


def _episode(job):
    arm_path, base_path, scenario, folder, duration = job
    import contextlib

    import torch

    torch.set_num_threads(1)
    from comp0245.pipeline import read_json
    from comp0245.sim.obstacle_env import rollout

    config = read_json(SETTINGS_FILE)
    config["duration_s"] = duration
    policy = None
    if arm_path != "expert":
        base = None
        if base_path:
            from comp0245.evaluation.eval_obstacles import LearnedBase, load_model

            base = LearnedBase(*load_model(base_path))
        policy = LearnedArm(*load_arm(arm_path), scenario["target"], base)
    with open(os.devnull, "w") as quiet, contextlib.redirect_stderr(quiet):
        summary = rollout(config, scenario, folder, policy, save_every=10)
    reach_from = summary["base_arrival_time_s"] if policy is None else policy.reach_from
    return {
        "id": scenario["id"],
        "reached": summary["reached_target"],
        "contact_steps": summary["contact_steps"],
        "success": summary["reached_target"] and summary["contact_steps"] == 0,
        "final_error_m": summary["final_error_m"],
        "minimum_error_m": summary["minimum_error_m"],
        "parked_at_s": reach_from,
        "reach_time_s": (summary["simulation_time_s"] - reach_from)
        if reach_from is not None
        else None,
        "target_z_m": scenario["target"][2],
        "arm_start_changed": "arm_q0" in scenario,
    }


def scenarios():
    """The arm's held-out episodes (labelled by dataset) and 30 fresh normal scenarios."""
    sets = {}
    for folder, file in ARM_DATA.items():
        episodes = json.loads((ROOT / "runs" / folder / "manifest.json").read_text())["episodes"]
        held_out = {
            e["scenario_id"] for i, e in enumerate(episodes) if i % 10 == 0 and e["success"]
        }
        sets[folder] = [
            s for s in json.loads(scenario_path(file).read_text()) if s["id"] in held_out
        ]
    fresh = [
        s
        for s in json.loads(scenario_path("scenarios_obstacles_val100.json").read_text())
        if s["split"] == "validation"
    ]
    sets["fresh-normal"] = fresh[30:60]
    return sets


def run_eval(arm, base, root, workers=4, duration=40.0, limit=0):
    """Run every evaluation scenario with this arm (and base) into root; write results.json;
    return the table."""
    root.mkdir(parents=True)
    jobs, keys = [], []
    for label, chosen in scenarios().items():
        for s in chosen[: limit or None]:
            jobs.append((arm, base, s, str(root / label / s["id"]), duration))
            keys.append(label)
    print(f"{len(jobs)} episodes, {workers} workers", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool:
        results = pool.map(_episode, jobs, chunksize=1)
    table = []
    for label in dict.fromkeys(keys):
        rs = [r for k, r in zip(keys, results, strict=False) if k == label]
        times = [r["reach_time_s"] for r in rs if r["reached"] and r["reach_time_s"] is not None]
        table.append(
            {
                "set": label,
                "episodes": len(rs),
                "success": sum(r["success"] for r in rs),
                "reached": sum(r["reached"] for r in rs),
                "contact": sum(r["contact_steps"] > 0 for r in rs),
                "median_final_error_m": float(np.median([r["final_error_m"] for r in rs])),
                "worst_final_error_m": float(max(r["final_error_m"] for r in rs)),
                "median_reach_time_s": float(np.median(times)) if times else None,
            }
        )
        t = table[-1]
        print(
            f"{label:17s} success {t['success']}/{t['episodes']}, reached {t['reached']}, contact "
            f"{t['contact']}, "
            f"final error median {t['median_final_error_m'] * 1000:.1f} mm, worst "
            f"{t['worst_final_error_m'] * 1000:.0f} mm, "
            f"reach time median {t['median_reach_time_s']}",
            flush=True,
        )
    (root / "results.json").write_text(
        json.dumps(
            {
                "arm": arm,
                "base": base or "expert wheels",
                "duration_s": duration,
                "table": table,
                "episodes": [dict(set=k, **r) for k, r in zip(keys, results, strict=False)],
            },
            indent=1,
        )
    )
    return table


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", required=True, help="results go to runs/arm-eval-NAME")
    parser.add_argument(
        "--arm", required=True, help="arm model.pt, or 'expert' for the full-task expert"
    )
    parser.add_argument("--base", help="learned base model.pt (default: the expert's wheels)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument(
        "--limit", type=int, default=0, help="only the first N scenarios per set (smoke tests)"
    )
    args = parser.parse_args()
    root = ROOT / "runs" / f"arm-eval-{args.name}"
    if root.exists():
        raise FileExistsError(f"Choose a new name: {root}")
    run_eval(args.arm, args.base, root, args.workers, args.duration, args.limit)


if __name__ == "__main__":
    main()
