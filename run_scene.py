"""Run our room robot on scenario files, optionally in the PyBullet GUI: the room base (lidar, 72 rays, 3 scans) drives
with the arm folded, and once it passes the parking rule the arm unfolds and the room arm reaches (eval_arm.LearnedArm).
Used by the "Run ours" button of Julia's interactive_test.py (5 Nov) for demos; works on her scenes and ours.
One dot only: with several dots (scenario "targets") the first is used.

Example:
    python run_scene.py --scenarios ../comp0245_julia/runs/interactive/my_test.json --out runs/scenes --gui
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = "weights/base_rooms.pt"     # the room base (copy of runs/crossing-hold/r3-hold-latch/model.pt)
ARM = "weights/arm_rooms.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", required=True, help="JSON list of scenarios")
    parser.add_argument("--out", required=True, help="one folder per run goes here")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--duration", type=float, default=100.0)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--arm", default=ARM)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    from eval_arm import LearnedArm, load_arm
    from eval_obstacles import ARM_FOLD, LearnedBase, load_model
    from obstacle_env import rollout
    from pipeline import read_json
    config = read_json(ROOT/"settings.json")
    config["duration_s"] = args.duration
    model, saved = load_model(ROOT/args.base)
    config["lidar_rays"] = saved.get("rays", 36)
    config["lidar_range"] = saved.get("lidar_range", 4.0)
    arm = load_arm(ROOT/args.arm)
    for s in json.loads(Path(args.scenarios).read_text()):
        if len(s.get("targets", [])) > 1:
            print(f"{s['id']}: our robot does one dot; using the first of {len(s['targets'])}")
            s = {k: v for k, v in s.items() if k not in ("targets", "parks")}
        s = {**s, "arm_q0": ARM_FOLD}
        policy = LearnedArm(*arm, s["target"], LearnedBase(model, saved), unfold=True)
        summary = rollout(config, s, Path(args.out)/s["id"], policy, gui=args.gui, save_every=10)
        print(f"{s['id']}: {summary['termination']}, clean {summary['success']}, contact steps {summary['contact_steps']}, "
              f"parked at {policy.parked_at}, {summary['simulation_time_s']:.1f} s simulated", flush=True)


if __name__ == "__main__":
    main()
