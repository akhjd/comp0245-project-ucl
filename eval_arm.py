"""Closed-loop test of a learned arm: does the hand actually reach the target when the network controls the arm?

The expert's wheels drive and park (same logic as eval_obstacles.ExpertHold), so a miss is the arm's fault and
not bad parking. While driving, the arm holds its start pose like the expert does (PD + gravity). Once the base
has parked, the arm torque is
    gravity_torque(q) + network(arm_inputs)          (train_arm.py)
The episode ends with the course's success rule (hand within 1 cm for 0.25 s), or after --duration.
--base MODEL.pt drives with a learned base instead (the fully learned robot); the environment gets the model's own
number of lidar rays. The arm takes over only once the base passes the same rule as the reported "parked" score:
inside the reach area, facing the target and stopped, for the whole last SETTLE_WINDOW_S (5 Nov; was: still for 1 s
anywhere). --arm-fold drives with the arm folded (eval_obstacles.ARM_FOLD, as the room models were trained); after
the hand-off the arm first unfolds to the normal start pose, which the learned arm was trained from.

Scenarios: the arm's held-out episodes (every 10th successful episode of each dataset, never trained on) and
validationc_030-059 (fresh, normal heights). --fresh A B picks other fresh scenarios and --no-held-out drops the
held-out set (e.g. the final full-robot check on validationc_060-099, never used to choose anything).
The test split is used only with --split test (the single final evaluation, 5 Nov).

Example:
    python eval_arm.py --name arm-a --arm runs/arm-a/model.pt
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
ARM_DATA = {"obst-v4-height": "scenarios_obstacles_v4_height.json",
            "obst-v4-armstart": "scenarios_obstacles_v4_armstart.json",
            "obst-v4-both": "scenarios_obstacles_v4_both.json"}
KP, KD = 100.0, 30.0       # the expert's arm stiffness
# The normal arm start pose (no scenario arm_q0); the learned arm's training episodes started within +-0.3 rad of it
ARM_START = [0.0, 1.57, 0.0, 1.0, 0.0, 1.0, 0.0]
# Unfolding after a folded drive: the hold pose ramps in a straight joint-space line to ARM_START at the expert's own
# retract rate; the learned arm starts once the joints have settled. The straight path keeps >= 8.3 cm from the robot's
# body but swings up to 0.22 m outside the base footprint (runs/arm-tuck probe, 5 Nov)
UNFOLD_RATE, UNFOLD_SETTLED = 0.5, 0.05   # rad/s


def load_arm(path):
    """Rebuild the arm network from model.pt (same layers as train_arm.py)."""
    import torch
    from torch import nn
    from arm_data import arm_input_fn
    saved = torch.load(path, map_location="cpu", weights_only=False)
    arm_input_fn(saved["input_names"])     # model A (24 inputs) or model B (31); anything else is refused
    h, n = saved["hidden"], len(saved["input_names"])
    model = nn.Sequential(nn.Linear(n, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 7))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved


class LearnedArm:
    """policy.act(state) -> 11 commands. Base: expert wheels (or a learned base); arm: hold, then learned reach.
    Make a new one for every episode."""
    def __init__(self, arm, arm_saved, target, base=None, unfold=False):
        from eval_obstacles import ExpertHold
        self.arm, self.s = arm, arm_saved
        self.target = np.asarray(target, dtype=float)
        self.base = base or ExpertHold(target)
        self.q_hold = None
        self.unfold = unfold       # arm folded while driving: unfold to ARM_START before the learned arm takes over
        self.parked_at = None      # time (s) the base passed the parking rule
        self.reach_from = None     # time (s) the arm started reaching
        self.t, self.still = 0.0, 0.0

    def parked(self, state):
        """Expert wheels: when the expert says it has arrived. Learned base: the "parked" rule of eval_obstacles.score
        (reach area with the facing, facing limit, stopped) has held for the last SETTLE_WINDOW_S."""
        from eval_obstacles import reach_limit, FACING_LIMIT, MAX_SPEED, MAX_YAW_RATE, REACH_MIN_M, SETTLE_WINDOW_S
        from obstacle_env import wrap
        if hasattr(self.base, "arrived"):
            return self.base.arrived
        to_target = self.target[:2] - state["base"][:2]
        distance = float(np.hypot(*to_target))
        facing = abs(wrap(np.arctan2(to_target[1], to_target[0]) - state["yaw"]))
        stopped = np.hypot(*state["x"][22:24]) < MAX_SPEED and abs(state["x"][27]) < MAX_YAW_RATE
        ok = stopped and facing <= FACING_LIMIT and REACH_MIN_M <= distance <= reach_limit(self.target[2], facing)
        self.still = self.still + 0.001 if ok and self.t >= 2.0 else 0.0
        return self.still >= SETTLE_WINDOW_S

    def unfolding(self, qd):
        """Ramp the hold pose towards ARM_START; True until it is there and the joints have settled."""
        step = np.clip(np.array(ARM_START) - self.q_hold, -UNFOLD_RATE*0.001, UNFOLD_RATE*0.001)
        self.q_hold = self.q_hold + step
        return np.abs(step).max() > 0 or np.abs(qd).max() >= UNFOLD_SETTLED

    def act(self, state):
        import torch
        from arm_data import gravity_torque, arm_input_fn
        q, qd = state["q"][4:11], state["qd"][4:11]
        if self.q_hold is None:
            self.q_hold = q.copy()
        wheels = self.base.act(state)[:4]
        if self.parked_at is None and self.parked(state):
            self.parked_at = self.t
        if self.parked_at is not None and self.reach_from is None:
            wheels = np.zeros(4)
            if not (self.unfold and self.unfolding(qd)):
                self.reach_from = self.t
        if self.reach_from is None:     # driving (or unfolding): hold the pose like the expert (PD + gravity)
            arm = KP*(self.q_hold - q) - KD*qd + gravity_torque(q)
        else:
            wheels = np.zeros(4)
            x = (arm_input_fn(self.s["input_names"])(state["x"], state["ee"]) - self.s["x_mean"]) / self.s["x_std"]
            with torch.no_grad():
                out = self.arm(torch.from_numpy(x.astype(np.float32))[None])[0].numpy()
            arm = gravity_torque(q) + out*self.s["y_std"] + self.s["y_mean"]
        self.t += 0.001
        return np.concatenate([wheels, arm])


def _episode(job):
    """job: (arm, base, scenario, folder, duration[, perturb]); perturb as in obstacle_env.rollout (robust_full.py)."""
    arm_path, base_path, scenario, folder, duration, *perturb = job
    from eval_obstacles import ARM_FOLD
    import contextlib
    import torch
    torch.set_num_threads(1)
    from obstacle_env import rollout
    from pipeline import read_json
    config = read_json(ROOT/"settings.json")
    config["duration_s"] = duration
    policy = None
    if arm_path != "expert":
        base = None
        if base_path:
            from eval_obstacles import load_model, LearnedBase
            model, saved = load_model(base_path)
            config["lidar_rays"] = saved.get("rays", 36)    # the model's own (LearnedBase refuses any other count)
            config["lidar_range"] = saved.get("lidar_range", 4.0)
            base = LearnedBase(model, saved)
        policy = LearnedArm(*load_arm(arm_path), scenario["target"], base, unfold=scenario.get("arm_q0") == ARM_FOLD)
    with open(os.devnull, "w") as quiet, contextlib.redirect_stderr(quiet):
        summary = rollout(config, scenario, folder, policy, save_every=10, perturb=perturb[0] if perturb else None)
    reach_from = summary["base_arrival_time_s"] if policy is None else policy.reach_from
    return {"id": scenario["id"], "reached": summary["reached_target"], "contact_steps": summary["contact_steps"],
            "success": summary["reached_target"] and summary["contact_steps"] == 0,
            "final_error_m": summary["final_error_m"], "minimum_error_m": summary["minimum_error_m"],
            "parked_at_s": reach_from if policy is None else policy.parked_at,
            "unfold_s": (reach_from - policy.parked_at) if policy is not None and reach_from is not None else None,
            "reach_time_s": (summary["simulation_time_s"] - reach_from) if reach_from is not None else None,
            "target_z_m": scenario["target"][2], "arm_start_changed": "arm_q0" in scenario}


def scenarios(fresh=(30, 60), held_out=True, fresh_file="scenarios_obstacles_val100.json", split="validation"):
    """The arm's held-out episodes (labelled by dataset) and fresh normal scenarios fresh_file[fresh] of this split."""
    if split != "validation" and held_out:
        raise ValueError("the arm's held-out episodes are validation data: use --no-held-out with --split test")
    sets = {}
    for folder, file in (ARM_DATA.items() if held_out else ()):
        episodes = json.loads((ROOT/"runs"/folder/"manifest.json").read_text())["episodes"]
        held_out = {e["scenario_id"] for i, e in enumerate(episodes) if i % 10 == 0 and e["success"]}
        sets[folder] = [s for s in json.loads((ROOT/file).read_text()) if s["id"] in held_out]
    fresh_all = [s for s in json.loads((ROOT/fresh_file).read_text()) if s["split"] == split]
    sets["fresh-normal" if split == "validation" else split] = fresh_all[slice(*fresh)]
    return sets


def run_eval(arm, base, root, workers=4, duration=40.0, limit=0, fresh=(30, 60), held_out=True,
             fresh_file="scenarios_obstacles_val100.json", arm_fold=False, split="validation"):
    """Run every evaluation scenario with this arm (and base) into root; write results.json; return the table."""
    from eval_obstacles import ARM_FOLD
    root.mkdir(parents=True)
    jobs, keys = [], []
    for label, chosen in scenarios(fresh, held_out, fresh_file, split).items():
        for s in chosen[:limit or None]:
            if arm_fold:
                s = {**s, "arm_q0": ARM_FOLD}
            jobs.append((arm, base, s, str(root/label/s["id"]), duration))
            keys.append(label)
    print(f"{len(jobs)} episodes, {workers} workers", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool:
        results = pool.map(_episode, jobs, chunksize=1)
    table = []
    for label in dict.fromkeys(keys):
        rs = [r for k, r in zip(keys, results) if k == label]
        times = [r["reach_time_s"] for r in rs if r["reached"] and r["reach_time_s"] is not None]
        table.append({"set": label, "episodes": len(rs), "success": sum(r["success"] for r in rs),
                      "reached": sum(r["reached"] for r in rs), "contact": sum(r["contact_steps"] > 0 for r in rs),
                      "median_final_error_m": float(np.median([r["final_error_m"] for r in rs])),
                      "worst_final_error_m": float(max(r["final_error_m"] for r in rs)),
                      "median_reach_time_s": float(np.median(times)) if times else None})
        t = table[-1]
        print(f"{label:17s} success {t['success']}/{t['episodes']}, reached {t['reached']}, contact {t['contact']}, "
              f"final error median {t['median_final_error_m']*1000:.1f} mm, worst {t['worst_final_error_m']*1000:.0f} mm, "
              f"reach time median {t['median_reach_time_s']}", flush=True)
    (root/"results.json").write_text(json.dumps({"arm": arm, "base": base or "expert wheels",
        "duration_s": duration, "arm_fold": arm_fold, "fresh_file": fresh_file, "split": split, "table": table,
        "episodes": [dict(set=k, **r) for k, r in zip(keys, results)]}, indent=1))
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="results go to runs/arm-eval-NAME")
    parser.add_argument("--arm", required=True, help="arm model.pt, or 'expert' for the full-task expert")
    parser.add_argument("--base", help="learned base model.pt (default: the expert's wheels)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--limit", type=int, default=0, help="only the first N scenarios per set (smoke tests)")
    parser.add_argument("--fresh", type=int, nargs=2, default=[30, 60], help="which validationc_ scenarios (from, to)")
    parser.add_argument("--no-held-out", action="store_true", help="skip the arm's held-out episodes")
    parser.add_argument("--fresh-file", default="scenarios_obstacles_val100.json", help="file of fresh scenarios")
    parser.add_argument("--arm-fold", action="store_true", help="drive with the arm folded, unfold before reaching")
    parser.add_argument("--split", default="validation", choices=["validation", "test"],
                        help="test only for the single final evaluation")
    args = parser.parse_args()
    root = ROOT/"runs"/f"arm-eval-{args.name}"
    if root.exists():
        raise FileExistsError(f"Choose a new name: {root}")
    run_eval(args.arm, args.base, root, args.workers, args.duration, args.limit, tuple(args.fresh),
             not args.no_held_out, args.fresh_file, args.arm_fold, args.split)


if __name__ == "__main__":
    main()
