"""Follow-up to oracle_check.py: knowing where the expert will be 1-2 s later makes its wheel commands easy to copy.
Can that waypoint itself be predicted from the real inputs (3 scans, no velocity)? Same 2 x 64 fit; target = the
4 waypoint numbers (forward, left, sin and cos of the heading change) instead of the wheels. Saved data only."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from why_no_learning import rows, r2, NO_VEL
from oracle_check import intent

OUT = HERE/"waypoint_check.json"


def fit(x, y, xv, yv, near, epochs=12):
    torch.manual_seed(0)
    mx, sx, my, sy = x.mean(0), x.std(0) + 1e-6, y.mean(0), y.std(0) + 1e-6
    X, Y, XV = torch.tensor((x - mx)/sx), torch.tensor((y - my)/sy), torch.tensor((xv - mx)/sx)
    model = nn.Sequential(nn.Linear(x.shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, y.shape[1]))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = {"all": -9.0}
    for _ in range(epochs):
        order = torch.randperm(len(X))
        for start in range(0, len(X), 256):
            i = order[start:start+256]
            loss = ((model(X[i]) - Y[i])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = model(XV).numpy()*sy + my
        score = {"all": round(r2(pred, yv), 3), "mover_within_1m": round(r2(pred[near], yv[near]), 3),
                 "position_error_median_m": round(float(np.median(np.linalg.norm(pred[:, :2] - yv[:, :2], axis=1))), 3)}
        if score["all"] > best["all"]:
            best = score
    return best


if __name__ == "__main__":
    torch.set_num_threads(6)
    tr, va = rows("runs/rooms-expert", "train"), rows("runs/rooms-expert-val", "validation")
    x, xv = np.hstack([tr["x"][:, NO_VEL], tr["lidar3"][:, 36:]]), np.hstack([va["x"][:, NO_VEL], va["lidar3"][:, 36:]])
    near = va["near"] < 1.0
    results = {}
    for name, ahead in (("waypoint1s", 100), ("waypoint2s", 200)):
        yi, yvi = intent("runs/rooms-expert", "train", ahead), intent("runs/rooms-expert-val", "validation", ahead)
        results[name] = fit(x, yi, xv, yvi, near)
        results[name + "_with_true_movers"] = fit(np.hstack([x, tr["mover"]]), yi, np.hstack([xv, va["mover"]]), yvi, near)
        print(name, results[name], results[name + "_with_true_movers"], flush=True)
        OUT.write_text(json.dumps(results, indent=1))
