"""Closed-loop test of the arm MLP: the expert drives and parks, then the MLP's torques reach for
the dot.

    python -m comp0245.evaluation.eval_arm_mlp --arm runs/arm1/best.pt --out runs/arm1-drive

The expert still decides when to reach, which dot to reach for, and handles holding and
retracting, so a miss is the arm's fault. Uses the validation scenarios only. Results go to
evaluation.json. Run it again to continue after an interruption.
"""

import argparse
import collections
import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path

import numpy as np

from comp0245.paths import SETTINGS_FILE
from comp0245.training.train_mlp import CASES

REACH_LIMIT_S = 30.0  # no base movement or touched dot for this long ends the run


class LearnedReach:
    """Torques from the MLP while the expert's arm would be reaching, clipped per joint; None
    otherwise. New one per episode."""

    def __init__(self, model, saved):
        self.model, self.s = model, saved
        self.rows, self.clipped = 0, 0

    def __call__(self, state, robot, elapsed):
        if not robot.arrived or robot.retracting:
            return None
        import torch

        from comp0245.arm_data import gravity_torque
        from comp0245.models import arm_inputs

        s = self.s
        x = (arm_inputs(state["x"], state["ee"]) - s["x_mean"]) / s["x_std"]
        with torch.no_grad():
            out = self.model(torch.from_numpy(x.astype(np.float32))[None])[0].numpy()
        torque = gravity_torque(state["q"][4:11]) + out * s["y_std"] + s["y_mean"]
        self.rows += 1
        self.clipped += int(np.any(np.abs(torque) > s["limit"]))
        return np.clip(torque, -s["limit"], s["limit"])


def drive(job):
    model_path, scenario, folder, duration = job
    import torch

    from comp0245.pipeline import read_json
    from comp0245.sim.obstacle_env import rollout
    from comp0245.training.train_arm_mlp import load_arm

    torch.set_num_threads(1)
    done = Path(folder) / "summary.json"
    if done.exists():
        return json.loads(done.read_text())
    shutil.rmtree(folder, ignore_errors=True)
    config = read_json(SETTINGS_FILE)
    config["duration_s"] = duration
    arm = LearnedReach(*load_arm(model_path))
    summary = rollout(
        config, scenario, folder, save_every=10, learned_arm=arm, stall_limit_s=REACH_LIMIT_S
    )
    summary.update(arm_rows=arm.rows, arm_rows_clipped=arm.clipped)
    done.write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default="runs/dataset")
    parser.add_argument("--only", nargs="*", help="scenario IDs to run")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    from comp0245.training.dagger_mlp import scenarios_of

    scenarios = [s for v in scenarios_of(args.data, "validation").values() for s in v]
    if args.only:
        scenarios = [s for s in scenarios if s["id"] in args.only]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    begin = time.time()
    jobs = [(str(args.arm), s, str(out / s["id"]), args.duration) for s in scenarios]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(drive, jobs, chunksize=1)

    def tally(rows):
        good = [r for r in rows if r["success"]]
        return {
            "runs": len(rows),
            "success": len(good),
            "dots_touched": sum(r["dots_picked"] for r in rows),
            "dots_total": sum(r["dots_total"] for r in rows),
            "any_contact": sum(r["contact_steps"] > 0 for r in rows),
            "ended": dict(collections.Counter(r["termination"] for r in rows)),
            "clipped_share": sum(r["arm_rows_clipped"] for r in rows)
            / max(sum(r["arm_rows"] for r in rows), 1),
        }

    scores = {
        "all": tally(results),
        "seconds": time.time() - begin,
        **{
            c: tally([r for r, s in zip(results, scenarios, strict=False) if s["case"] == c])
            for c in CASES
        },
    }
    (out / "evaluation.json").write_text(json.dumps(scores, indent=1))
    a = scores["all"]
    print(
        f"ARM DRIVING: success {a['success']}/{a['runs']}, dots "
        f"{a['dots_touched']}/{a['dots_total']}, ended {a['ended']}, "
        f"torque beyond limits {100 * a['clipped_share']:.2f}% of reach steps\n  "
        + ", ".join(f"{c} {scores[c]['success']}/{scores[c]['runs']}" for c in CASES),
        flush=True,
    )


if __name__ == "__main__":
    main()
