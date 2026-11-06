"""Write a reproducible scenarios file with random targets, for `run.sh collect --scenarios FILE`.

Targets are sampled in polar form around the robot's fixed start (world origin):
distance r from the origin, direction (bearing) and height z. Each split gets its own targets, so
validation/test targets never appear in training.

Example:
    python -m comp0245.scenarios.make_scenarios --out scenarios_random_v1.json --train 60 \\
        --validation 10 --test 10 --seed 0
"""

import argparse
import json
from pathlib import Path

import numpy as np


def sample_target(rng, radius, bearing_deg, height):
    """A target [x, y, z] (m, world frame) at a random distance, bearing (degrees) and height."""
    r = rng.uniform(*radius)
    bearing = np.radians(rng.uniform(*bearing_deg))
    z = rng.uniform(*height)
    return [
        round(float(r * np.cos(bearing)), 4),
        round(float(r * np.sin(bearing)), 4),
        round(float(z), 4),
    ]


def main():
    """Command-line entry point: write obstacle-free train/validation/test scenarios to --out."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--train", type=int, default=60)
    parser.add_argument("--validation", type=int, default=10)
    parser.add_argument("--test", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    # Defaults cover the pilot targets (r 0.9-1.3 m, z 0.45-0.6 m) with a wider margin.
    parser.add_argument(
        "--radius", type=float, nargs=2, default=[0.8, 2.0], help="metres from origin"
    )
    parser.add_argument("--bearing-deg", type=float, nargs=2, default=[-180.0, 180.0])
    parser.add_argument("--height", type=float, nargs=2, default=[0.35, 0.8], help="metres")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Choose a new scenarios file: {out}")
    rng = np.random.default_rng(args.seed)
    scenarios = []
    for split, count in [
        ("train", args.train),
        ("validation", args.validation),
        ("test", args.test),
    ]:
        for i in range(count):
            scenarios.append(
                {
                    "id": f"{split}_{i:03d}",
                    "split": split,
                    "target": sample_target(rng, args.radius, args.bearing_deg, args.height),
                    "seed": int(rng.integers(1_000_000)),
                }
            )
    out.write_text(json.dumps(scenarios, indent=1))
    print(f"Wrote {len(scenarios)} scenarios to {out}")


if __name__ == "__main__":
    main()
