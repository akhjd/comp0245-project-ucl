"""Is the steady expert (SteadyRRTExpert: near-ties settled by side) easier to copy than the visible RRT expert?
Same 2 x 64 fit and inputs (3 scans, no velocity) as oracle_check.py; saved expert data only."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from why_no_learning import rows, NO_VEL, jumpiness
from oracle_check import fit

if __name__ == "__main__":
    torch.set_num_threads(6)
    out = {}
    for name, train, val in (("visible", "runs/rooms-expert", "runs/rooms-expert-val"),
                             ("steady", "runs/rooms-expert-steady", "runs/rooms-expert-steady-val")):
        tr, va = rows(train, "train"), rows(val, "validation")
        x, xv = np.hstack([tr["x"][:, NO_VEL], tr["lidar3"][:, 36:]]), np.hstack([va["x"][:, NO_VEL], va["lidar3"][:, 36:]])
        succ = [e["success"] for e in json.loads((Path("/home/akhjd/ros2_ws/comp0245_project")/val/"manifest.json").read_text())["episodes"]]
        out[name] = {"expert_val_success": f"{sum(succ)}/{len(succ)}", "jumps": jumpiness(va), **fit(x, tr["y"], xv, va["y"], va["near"] < 1.0)}
        print(name, out[name], flush=True)
    (HERE/"steady_copy_check.json").write_text(json.dumps(out, indent=1))
