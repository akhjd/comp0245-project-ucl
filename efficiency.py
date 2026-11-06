"""Efficiency of the final models: parameter count, file size and CPU time per control step (1 thread).

Per-step time is measured on states from a saved fully learned episode: the base network alone, the arm's
inputs + gravity torque (pinocchio) + network, and the whole act() as recorded in the run (inference_ms, which
includes everything the policy does in that step). The control loop runs at 1 kHz, so a step must take < 1 ms.

Example:
    python efficiency.py --run runs/robust-full/fully_learned/clean --out runs/efficiency.json
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def per_call_us(fn, inputs, repeat=3):
    """Median over repeats of the mean time per call, microseconds."""
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        for x in inputs:
            fn(x)
        times.append((time.perf_counter() - start) / len(inputs) * 1e6)
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="weights/base_final.pt")
    parser.add_argument("--arm", default="weights/arm_final.pt")
    parser.add_argument("--run", required=True, help="folder of fully learned episodes (trajectory.npz each)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    from arm_data import gravity_torque
    from eval_arm import load_arm
    from eval_obstacles import load_model
    from models import arm_inputs, base_inputs

    base, bs = load_model(ROOT/args.base)
    arm, arms = load_arm(ROOT/args.arm)
    episodes = sorted(p for p in (ROOT/args.run).iterdir() if (p/"trajectory.npz").exists())
    obs, ee, lidar, recorded = [], [], [], []
    for p in episodes:
        z = np.load(p/"trajectory.npz")
        obs.append(z["observations"]); ee.append(z["ee_m"]); lidar.append(z["lidar"]); recorded.append(z["inference_ms"])
    obs, ee, lidar, recorded = (np.concatenate(a) for a in (obs, ee, lidar, recorded))
    pick = np.random.default_rng(0).choice(len(obs), size=min(2000, len(obs)), replace=False)

    # The model's own input layout (5 Nov): its number of rays and scans. The older scans repeat the current one
    # here: the time per step does not depend on the values
    scans, rays = bs.get("scans", 1), bs.get("rays", 36)
    if lidar.shape[1] != rays:
        raise ValueError(f"the base model uses {rays} lidar rays but the run saved {lidar.shape[1]}")
    hold = getattr(base, "hold", None)

    def base_step(i):
        x = np.hstack([base_inputs(obs[i])] + [lidar[i]]*scans).astype(np.float32)
        if bs.get("columns") is not None:
            x = x[bs["columns"]]
        with torch.no_grad():
            xt = torch.from_numpy((x - bs["x_mean"]) / bs["x_std"])[None]
            if hold is not None:          # wheels and the hold probability from the same features, as LearnedBase
                h = base.features(xt)
                base.wheels(h), hold(h)
            else:
                base(xt)

    def arm_step(i):
        x = (arm_inputs(obs[i], ee[i]) - arms["x_mean"]) / arms["x_std"]
        with torch.no_grad():
            arm(torch.from_numpy(x.astype(np.float32))[None])
        gravity_torque(obs[i][8:15])

    count = lambda m: int(sum(p.numel() for p in m.parameters()))
    result = {
        "base": {"parameters": count(base), "file_bytes": (ROOT/args.base).stat().st_size,
                 "inputs": len(bs["input_names"]), "scans": scans, "rays": rays, "hold": hold is not None,
                 "step_us": per_call_us(base_step, pick)},
        "arm": {"parameters": count(arm), "file_bytes": (ROOT/args.arm).stat().st_size,
                "inputs": len(arms["input_names"]), "step_us": per_call_us(arm_step, pick)},
        "recorded_act_ms": {"episodes": len(episodes), "median": float(np.median(recorded)),
                            "p99": float(np.percentile(recorded, 99)), "max": float(recorded.max())},
        "cpu": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?"),
        "torch": torch.__version__, "threads": 1,
    }
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
