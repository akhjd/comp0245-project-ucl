"""Network size check (4 Nov): is the room model too small to copy the dynamic window teacher? Same network as the
window DAgger (ray filter + threat, 3 scans, no velocity, 72 rays) at 64 / 128 / 256 neurons per layer, two seeds,
plain cloning on runs/rooms-window72 (successful runs), scored on rooms-window72-val. R2 of the wheel commands on
train and validation rows, all rows and rows with a mover within 1 m. Offline only; no DAgger, no test split.

Rule written before the results (AGENT_LOG 4 Nov): validation R2 near movers up by >= 0.1 over 64 (mean of seeds)
-> run DAgger with the bigger network; only train R2 up -> memorising, size is not the fix."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")    # runs/ is a symlink to /data, so not from __file__
sys.path.insert(0, str(ROOT))
from train_base_torch import train, load, STRIDE
from eval_obstacles import load_model

HERE = ROOT/"runs/scans-rooms/fit-check"
OUT = HERE/"size-check"
DATA, VAL = ROOT/"runs/rooms-window72", ROOT/"runs/rooms-window72-val"


def r2(pred, y):
    return float(1 - ((pred - y)**2).sum()/((y - y.mean(axis=0))**2).sum())


def near_movers(folder, split):
    """Per row of load(split, folder): is a moving obstacle within 1 m of the base? Same episodes, rows and order."""
    manifest = json.loads((folder/"manifest.json").read_text())
    stride = max(1, STRIDE // manifest.get("saved_every", 1))
    out = []
    for ep in manifest["episodes"]:
        if ep["split"] != split or not ep["success"]:
            continue
        z = np.load(folder/ep["path"]/"trajectory.npz")
        moving = np.array(ep.get("obstacle_is_moving", []), bool)
        n = len(z["observations"][::stride])
        if not moving.any():
            out.append(np.zeros(n, bool))
            continue
        xy = z["obstacle_xy"][::stride][:, moving]
        out.append(np.linalg.norm(xy - z["base_m"][::stride][:, None, :2], axis=2).min(axis=1) < 1.0)
    return np.concatenate(out)


def scores(model_path, rows):
    model, saved = load_model(model_path)
    out = {}
    for name, (x, y, near) in rows.items():
        xs = ((x[:, saved["columns"]] - saved["x_mean"])/saved["x_std"]).astype(np.float32)
        with torch.no_grad():
            pred = np.concatenate([model(torch.from_numpy(xs[i:i + 65536])).numpy() for i in range(0, len(xs), 65536)])
        pred = pred*saved["y_std"] + saved["y_mean"]
        out[name] = {"all": round(r2(pred, y), 3), "mover_within_1m": round(r2(pred[near], y[near]), 3)}
    return out


if __name__ == "__main__":
    torch.set_num_threads(8)
    rows = {}
    for name, folder, split in (("train", DATA, "train"), ("validation", VAL, "validation")):
        x, y = load(split, folder, scans=3)[:2]
        rows[name] = (x, y, near_movers(folder, split))
        print(name, len(x), "rows,", f"{100*rows[name][2].mean():.1f}% near a mover", flush=True)
    results = json.loads((HERE/"size_check.json").read_text()) if (HERE/"size_check.json").exists() else {}
    for hidden in (64, 128, 256):
        for seed in (0, 1):
            key = f"hidden{hidden}-seed{seed}"
            if key in results:
                continue
            rmse = train([str(DATA)], [], OUT/key, True, seed=seed, scans=3, hidden=hidden, encoder="conv",
                         threat=True, validation=str(VAL))
            info = json.loads((OUT/key/"training.json").read_text())
            results[key] = {"hidden": hidden, "seed": seed, "parameters": info["parameters"],
                            "best_epoch": info["best_epoch"], "validation_rmse": round(rmse, 3),
                            **scores(OUT/key/"model.pt", rows)}
            print(key, results[key], flush=True)
            (HERE/"size_check.json").write_text(json.dumps(results, indent=1))
