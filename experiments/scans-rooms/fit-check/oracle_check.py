"""Cheat-input check (3 Nov evening): which missing information stops the room models from copying the expert?

Same small network and training as why_no_learning.py (2 x 64, 12 epochs, squared error, best validation R2), on
the saved expert runs only (no new simulation, no test data). Each variant adds one piece of information the real
model never has:
  movers  true position and velocity of the 3 nearest movers, robot frame (perception cheat)
  intent  where the expert itself will be 1 s / 2 s later, robot frame (cheat for the plan it keeps in memory)
Control: the same intent cheat on the open-floor data, whose expert (SteerAround) has no plan memory. If intent
helps the rooms much more than the open floor, the hidden plan is what the model is missing; if it helps both
alike, it is just the future leaking the present command.
R2 is reported for all rows and for rows with a mover within 1 m, where the hits happen.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from why_no_learning import ROOT, rows, r2, NO_VEL

OUT = HERE/"oracle_check.json"
AHEAD = {"intent1s": 100, "intent2s": 200}     # saved rows are 10 ms apart after the stride
results = {}


def say(name, value):
    results[name] = value
    print(name, value, flush=True)
    OUT.write_text(json.dumps(results, indent=1))


def intent(folder, split, ahead):
    """Per kept row: the expert's own position `ahead` rows later in the robot's frame now (forward, left), plus
    its heading change. Same episode selection and order as why_no_learning.rows."""
    folder = ROOT/folder
    out = []
    for ep in json.loads((folder/"manifest.json").read_text())["episodes"]:
        if ep["split"] != split or not ep["success"]:
            continue
        z = np.load(folder/ep["path"]/"trajectory.npz")
        obs, base = z["observations"][::10], z["base_m"][::10]
        later = np.minimum(np.arange(len(obs)) + ahead, len(obs) - 1)
        yaw = np.arctan2(obs[:, 3], obs[:, 4])
        d = base[later, :2] - base[:, :2]
        c, s = np.cos(yaw), np.sin(yaw)
        turn = np.angle(np.exp(1j*(yaw[later] - yaw)))
        out.append(np.stack([c*d[:, 0] + s*d[:, 1], c*d[:, 1] - s*d[:, 0], np.sin(turn), np.cos(turn)], axis=1))
    return np.concatenate(out).astype(np.float32)


def fit(x, y, xv, yv, near, epochs=12, seed=0):
    """2 x 64 network; returns the best validation R2 over epochs, on all rows and on rows near a mover."""
    torch.manual_seed(seed)
    mx, sx, my, sy = x.mean(0), x.std(0) + 1e-6, y.mean(0), y.std(0) + 1e-6
    X, Y, XV = torch.tensor((x - mx)/sx), torch.tensor((y - my)/sy), torch.tensor((xv - mx)/sx)
    model = nn.Sequential(nn.Linear(x.shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = {"all": -9.0}
    for _ in range(epochs):
        order = torch.randperm(len(X))
        for start in range(0, len(X), 256):
            i = order[start:start+256]
            loss = ((model(X[i]) - Y[i])**2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = model(XV).numpy()*sy + my
        score = {"all": round(r2(pred, yv), 3)}
        if near is not None:
            score["mover_within_1m"] = round(r2(pred[near], yv[near]), 3)
        if score["all"] > best["all"]:
            best = score
    return best


if __name__ == "__main__":
    torch.set_num_threads(6)
    t = time.time()
    tr, va = rows("runs/rooms-expert", "train"), rows("runs/rooms-expert-val", "validation")
    x, xv = np.hstack([tr["x"][:, NO_VEL], tr["lidar3"][:, 36:]]), np.hstack([va["x"][:, NO_VEL], va["lidar3"][:, 36:]])
    near = va["near"] < 1.0
    cheats = {k: (intent("runs/rooms-expert", "train", a), intent("runs/rooms-expert-val", "validation", a)) for k, a in AHEAD.items()}
    print("loaded", len(x), len(xv), round(time.time() - t), "s", flush=True)
    say("rooms_rows", {"train": len(x), "validation": len(xv), "validation_mover_within_1m_pct": round(100*float(near.mean()), 1)})
    say("rooms_3scans", fit(x, tr["y"], xv, va["y"], near))
    say("rooms_3scans_movers", fit(np.hstack([x, tr["mover"]]), tr["y"], np.hstack([xv, va["mover"]]), va["y"], near))
    for k, (ci, cv) in cheats.items():
        say(f"rooms_3scans_{k}", fit(np.hstack([x, ci]), tr["y"], np.hstack([xv, cv]), va["y"], near))
    ci, cv = cheats["intent2s"]
    say("rooms_3scans_movers_intent2s", fit(np.hstack([x, tr["mover"], ci]), tr["y"], np.hstack([xv, va["mover"], cv]), va["y"], near))

    # Control: open floor, reactive expert without plan memory
    old_tr = [rows(f, "train") for f in ("runs/obst-v2b", "runs/obst-v3-extra-0", "runs/obst-v3-extra-1")]
    ox, oy = np.concatenate([d["x"] for d in old_tr])[:, NO_VEL], np.concatenate([d["y"] for d in old_tr])
    ov = rows("runs/obst-v2b", "validation")
    oi = np.concatenate([intent(f, "train", AHEAD["intent2s"]) for f in ("runs/obst-v2b", "runs/obst-v3-extra-0", "runs/obst-v3-extra-1")])
    oiv = intent("runs/obst-v2b", "validation", AHEAD["intent2s"])
    say("open_floor", fit(ox, oy, ov["x"][:, NO_VEL], ov["y"], None))
    say("open_floor_intent2s", fit(np.hstack([ox, oi]), oy, np.hstack([ov["x"][:, NO_VEL], oiv]), ov["y"], None))
    say("seconds", round(time.time() - t))
