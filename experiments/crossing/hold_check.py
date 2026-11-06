"""Offline check of a hold model (train_hold.py) on the expert validation rooms (runs/rooms-window72-val), 4 Nov.
Per row: the expert's command is exactly 0 ("hold"), split into "final" (from the expert's last non-zero command to
the end of the episode: parked for good) and "brief" (waiting for a mover, braking); driving rows split by whether a
mover is within 1.5 m. Usage: hold_check.py MODEL_FOLDER"""
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")
sys.path.insert(0, str(ROOT))
from eval_obstacles import load_model
from train_base_torch import load, STRIDE

VAL = ROOT/"runs/rooms-window72-val"


def row_info():
    """Per validation row, in load()'s order: final-hold flag and nearest mover distance (m)."""
    manifest = json.loads((VAL/"manifest.json").read_text())
    stride = max(1, STRIDE // manifest.get("saved_every", 1))
    final, near = [], []
    for ep in manifest["episodes"]:
        if ep["split"] != "validation" or not ep["success"]:
            continue
        z = np.load(VAL/ep["path"]/"trajectory.npz")
        y = z["actions"][::stride, :4]
        zero = (y == 0).all(axis=1)
        moving_rows = np.flatnonzero(~zero)
        last = moving_rows[-1] + 1 if len(moving_rows) else 0
        f = np.zeros(len(y), bool)
        f[last:] = True
        final.append(f)
        mv = np.array(ep["obstacle_is_moving"], bool)
        if mv.any():
            d = np.linalg.norm(z["obstacle_xy"][::stride][:, mv] - z["base_m"][::stride][:, None, :2], axis=2).min(axis=1)
        else:
            d = np.full(len(y), np.inf)
        near.append(d)
    return np.concatenate(final), np.concatenate(near)


if __name__ == "__main__":
    folder = ROOT/sys.argv[1]
    model, saved = load_model(folder/"model.pt")
    x, y = load("validation", VAL, scans=saved.get("scans", 1))[:2]
    x = ((x[:, saved["columns"]] - saved["x_mean"])/saved["x_std"]).astype(np.float32)
    with torch.no_grad():
        p = np.concatenate([torch.sigmoid(model.hold(model.features(torch.from_numpy(x[i:i + 65536]))))[:, 0].numpy()
                            for i in range(0, len(x), 65536)]) > 0.5
    zero = (y == 0).all(axis=1)
    final, near = row_info()
    assert len(final) == len(x)
    out = {"rows": int(len(x)), "hold_rows": int(zero.sum()), "final_rows": int(final.sum()),
           "recall_all_zero": float(p[zero].mean()), "recall_final": float(p[final].mean()),
           "recall_brief": float(p[zero & ~final].mean()) if (zero & ~final).any() else None,
           "false_hold_driving": float(p[~zero].mean()), "false_hold_driving_mover_1p5m": float(p[~zero & (near < 1.5)].mean()),
           "driving_mover_1p5m_rows": int((~zero & (near < 1.5)).sum())}
    print(json.dumps(out, indent=1))
    (folder/"hold_check.json").write_text(json.dumps(out, indent=1))
