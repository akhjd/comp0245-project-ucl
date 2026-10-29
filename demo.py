"""Watch one episode in the PyBullet viewer, at real-time speed, with the final models.

Modes:
    full    the fully learned robot: learned base parks, learned arm reaches
    base    learned base only: it parks, the arm holds its start pose (scored like eval_obstacles.py)
    arm     learned arm only: the expert's wheels park, the learned arm reaches (like eval_arm.py)
    expert  the full-task expert, for comparison

Examples:
    python demo.py full
    python demo.py base --scenario validationd_005
    python demo.py full --file scenarios_obstacles_hard2_gate.json --scenario validationhard2g_000
Each run is saved in runs/demo/ (its inference_ms includes the real-time waiting, so do not use it for timing).
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = "weights/base_final.pt"   # copy of runs/dagger-novnoise-hard/r3/model.pt (weights/README.md)
ARM = "weights/arm_final.pt"     # copy of runs/dagger-arm-b/r3/model.pt


class Paced:
    """Wraps a policy so the simulation runs no faster than `speed` x real time (1 control step = 1 ms)."""
    def __init__(self, policy, speed):
        self.policy, self.speed = policy, speed
        self.steps, self.start = 0, None

    def __getattr__(self, name):     # reach_from etc. come from the wrapped policy
        return getattr(self.policy, name)

    def act(self, state):
        if self.start is None:
            self.start = time.perf_counter()
        self.steps += 1
        ahead = self.steps*0.001/self.speed - (time.perf_counter() - self.start)
        if ahead > 0.01:
            time.sleep(ahead)
        return self.policy.act(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["full", "base", "arm", "expert"])
    parser.add_argument("--file", default="scenarios_obstacles_val_d.json", help="scenario file")
    parser.add_argument("--scenario", help="scenario id (default: the file's first)")
    parser.add_argument("--base", default=BASE, help="learned base model.pt")
    parser.add_argument("--arm", default=ARM, help="learned arm model.pt")
    parser.add_argument("--speed", type=float, default=1.0, help="x real time; 0 = as fast as possible")
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--headless", action="store_true", help="no viewer (for testing)")
    args = parser.parse_args()

    from eval_arm import LearnedArm, load_arm
    from eval_obstacles import LearnedBase, load_model, score
    from obstacle_env import rollout
    from pipeline import read_json
    scenarios = json.loads((ROOT/args.file).read_text())
    scenario = next(s for s in scenarios if s["id"] == args.scenario) if args.scenario else scenarios[0]
    if scenario["split"] == "test":
        raise ValueError("The test split is reserved for the final evaluation")
    config = read_json(ROOT/"settings.json")
    config["duration_s"] = args.duration

    policy = None
    if args.mode == "full":
        policy = LearnedArm(*load_arm(args.arm), scenario["target"], LearnedBase(*load_model(args.base)))
    elif args.mode == "base":
        policy = LearnedBase(*load_model(args.base))
    elif args.mode == "arm":
        policy = LearnedArm(*load_arm(args.arm), scenario["target"])
    if policy is not None and args.speed > 0:
        policy = Paced(policy, args.speed)
    folder = ROOT/"runs"/"demo"/f"{args.mode}-{scenario['id']}-{datetime.now():%H%M%S}"
    print(f"{args.mode}: {scenario['id']} ({len(scenario['obstacles'])} obstacles), target {scenario['target']}")
    summary = rollout(config, scenario, folder, policy, gui=not args.headless)

    print(f"ended: {summary['termination']} after {summary['simulation_time_s']:.1f} s, contact steps "
          f"{summary['contact_steps']}, final hand error {summary['final_error_m']*1000:.0f} mm")
    if args.mode == "base":
        s = score(folder, scenario, summary)
        print(f"parking: distance error {s['worst_distance_error_m']*100:.1f} cm, facing error "
              f"{s['worst_facing_error_rad']:.3f} rad, strict {s['strict']}, loose {s['loose']}")
    else:
        print("success" if summary["success"] else "no success")
    print(f"saved in {folder.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
