"""Summarise a data collection: python -m comp0245.evaluation.report_dataset runs/dataset. Lists
what was kept and dropped, per case."""

import glob
import json
import os
import sys

import numpy as np

folder = sys.argv[1]
total_mb = total_kept = 0
for case in ("case1", "case2", "case3", "case4", "case5"):
    runs = glob.glob(f"{folder}/{case}/part*/*/summary.json")
    if not runs:
        continue
    A = [json.load(open(f)) for f in runs]
    kept = [f for f in runs if os.path.exists(f.replace("summary.json", "trajectory.npz"))]
    mb = sum(os.path.getsize(f.replace("summary.json", "trajectory.npz")) for f in kept) / 1e6
    good = [a for a in A if a["success"]]
    total_mb += mb
    total_kept += len(kept)
    splits = [
        sum(
            json.load(open(f.replace("summary.json", "episode.json")))["scenario"]["split"] == x
            for f in kept
        )
        for x in ("train", "validation", "test")
    ]
    print(
        f"{case}: ran {len(A)}, kept {len(kept)} ({100 * len(kept) / len(A):.0f}%), timeouts "
        f"{sum(a['termination'] == 'timeout' for a in A)}, "
        f"contact {sum(a['contact_steps'] > 0 for a in A)}, kept time mean "
        f"{np.mean([a['simulation_time_s'] for a in good]) if good else 0:.1f} s "
        f"max {max([a['simulation_time_s'] for a in good], default=0):.1f} s, "
        f"{mb / max(len(kept), 1):.1f} MB/run, "
        f"splits {splits}"
    )
print(f"total kept {total_kept}, {total_mb / 1000:.2f} GB")
