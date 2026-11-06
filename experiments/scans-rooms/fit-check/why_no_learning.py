"""Why do the room models fit so badly? Read-only: fits nothing into runs/, touches no test data."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.neighbors import NearestNeighbors

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")
sys.path.insert(0, str(ROOT))
from eval_obstacles import load_model
from models import base_inputs
from train_base_torch import stack_scans

STRIDE, PAD = 10, [5.0, 0.0, 0.0, 0.0]
NO_VEL = [i for i in range(45) if i not in (6, 7, 8)]
OUT = Path(__file__).with_name("why_no_learning.json")
results = {}


def say(name, value):
    results[name] = value
    print(name, value, flush=True)
    OUT.write_text(json.dumps(results, indent=1))


def rows(folder, split):
    folder = ROOT/folder
    manifest = json.loads((folder/"manifest.json").read_text())
    out = {k: [] for k in ("x", "lidar3", "y", "mover", "near", "waiting", "target")}
    jumps = []
    for ep in manifest["episodes"]:
        if ep["split"] != split or not ep["success"]:
            continue
        z = np.load(folder/ep["path"]/"trajectory.npz")
        obs, lidar, y = z["observations"][::STRIDE], z["lidar"][::STRIDE], z["actions"][::STRIDE, :4]
        base = np.array([base_inputs(r) for r in obs])
        out["x"].append(np.hstack([base, lidar]))
        out["lidar3"].append(stack_scans(lidar, 3, 50))
        out["y"].append(y)
        out["target"].append(base[:, 2])
        jumps.append(np.abs(y[10:] - y[:-10]).max(axis=1))
        n = len(y)
        if "obstacle_xy" in z.files:
            moving = np.array(json.loads((folder/ep["path"]/"summary.json").read_text())["obstacle_is_moving"], bool)
            xy = z["obstacle_xy"][::STRIDE][:, moving]
            speed = np.gradient(xy, 0.01, axis=0) if len(xy) > 1 else np.zeros_like(xy)
            rel = xy - z["base_m"][::STRIDE][:, None, :2]
            yaw = np.arctan2(obs[:, 3], obs[:, 4])[:, None]
            c, s = np.cos(yaw), np.sin(yaw)
            robot_frame = np.stack([c*rel[..., 0] + s*rel[..., 1], c*rel[..., 1] - s*rel[..., 0],
                                    c*speed[..., 0] + s*speed[..., 1], c*speed[..., 1] - s*speed[..., 0]], axis=2)
            dist = np.linalg.norm(rel, axis=2)
            order = np.argsort(dist, axis=1)
            info = np.full((n, 12), 0.0)
            info[:] = PAD*3
            for row in range(n):
                k = order[row][:3]
                info[row, :4*len(k)] = robot_frame[row, k].ravel()
            out["mover"].append(info)
            out["near"].append(dist.min(axis=1) if moving.any() else np.full(n, 99.0))
            out["waiting"].append(z["waiting"][::STRIDE].astype(bool))
    data = {k: np.concatenate(v).astype(np.float32) for k, v in out.items() if v}
    data["jump"] = np.concatenate(jumps)
    return data


def r2(pred, y):
    return float(1 - ((pred - y)**2).sum()/((y - y.mean(axis=0))**2).sum())


def rmse(pred, y):
    return float(np.sqrt(np.mean((pred - y)**2)))


def fit(x, y, xv, yv, hidden, layers, epochs=12, seed=0):
    torch.manual_seed(seed)
    mx, sx, my, sy = x.mean(0), x.std(0) + 1e-6, y.mean(0), y.std(0) + 1e-6
    X, Y = torch.tensor((x - mx)/sx), torch.tensor((y - my)/sy)
    XV = torch.tensor((xv - mx)/sx)
    parts, size = [], x.shape[1]
    for _ in range(layers):
        parts += [nn.Linear(size, hidden), nn.ReLU()]
        size = hidden
    model = nn.Sequential(*parts, nn.Linear(size, 4))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = -9.0
    for _ in range(epochs):
        order = torch.randperm(len(X))
        for start in range(0, len(X), 256):
            i = order[start:start+256]
            loss = ((model(X[i]) - Y[i])**2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            best = max(best, r2(model(XV).numpy()*sy + my, yv))
    with torch.no_grad():
        sample = torch.randperm(len(X))[:100000]
        train = r2(model(X[sample]).numpy()*sy + my, y[sample.numpy()])
    return {"validation_r2": round(best, 3), "train_r2": round(train, 3),
            "parameters": sum(p.numel() for p in model.parameters())}


def neighbour_gap(x, y, xv, yv):
    """Copy the label of the most similar input from another run. If that fails, equal inputs have unequal answers."""
    rng = np.random.default_rng(0)
    pool = rng.choice(len(x), min(300000, len(x)), replace=False)
    query = rng.choice(len(xv), 3000, replace=False)
    m, s = x.mean(0), x.std(0) + 1e-6
    index = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=-1).fit((x[pool] - m)/s)
    dist, nearest = index.kneighbors((xv[query] - m)/s)
    return {"nearest_copy_r2": round(r2(y[pool][nearest[:, 0]], yv[query]), 3), "median_input_distance": round(float(np.median(dist)), 2)}


def jumpiness(data):
    scale = data["y"].std(0).mean()
    return {"label_std_rad_s": round(float(scale), 2), "rows_changing_over_1_std_in_0.1s_pct": round(100*float((data["jump"] > scale).mean()), 1)}


if __name__ == "__main__":
    torch.set_num_threads(8)
    t = time.time()
    old_train = [rows(f, "train") for f in ("runs/obst-v2b", "runs/obst-v3-extra-0", "runs/obst-v3-extra-1")]
    old_x = np.concatenate([d["x"] for d in old_train])[:, NO_VEL]
    old_y = np.concatenate([d["y"] for d in old_train])
    old_v = rows("runs/obst-v2b", "validation")
    new = rows("runs/rooms-expert", "train")
    new_v = rows("runs/rooms-expert-val", "validation")
    print("loaded", len(old_x), len(new["x"]), round(time.time() - t), "s", flush=True)
    x, xv = new["x"][:, NO_VEL], new_v["x"][:, NO_VEL]
    x3, xv3 = np.hstack([x, new["lidar3"][:, 36:]]), np.hstack([xv, new_v["lidar3"][:, 36:]])

    say("rows", {"old_train": len(old_x), "old_val": len(old_v["y"]), "rooms_train": len(x), "rooms_val": len(xv)})
    say("labels_old", jumpiness(old_v))
    say("labels_rooms", jumpiness(new_v))
    say("rows_mover_within_1.5m_pct", round(100*float((new["near"] < 1.5).mean()), 1))
    say("rows_mover_within_1m_pct", round(100*float((new["near"] < 1.0).mean()), 1))
    say("rows_expert_waiting_pct", round(100*float(new["waiting"].mean()), 1))

    say("neighbour_old", neighbour_gap(old_x, old_y, old_v["x"][:, NO_VEL], old_v["y"]))
    say("neighbour_rooms", neighbour_gap(x, new["y"], xv, new_v["y"]))

    model, saved = load_model(ROOT/"runs/scans-rooms/scans1-seed0/model.pt")
    groups = {"all": np.ones(len(xv), bool), "mover_within_1.5m": new_v["near"] < 1.5, "no_mover_within_1.5m": new_v["near"] >= 1.5,
              "expert_waiting": new_v["waiting"] > 0.5, "target_within_1m": new_v["target"] < 1.0}
    with torch.no_grad():
        for name, path in (("scans1", "runs/scans-rooms/scans1-seed0/model.pt"), ("final_base", "weights/base_final.pt")):
            model, saved = load_model(ROOT/path)
            scaled = (new_v["x"][:, saved["columns"]] - saved["x_mean"])/saved["x_std"]
            pred = model(torch.tensor(scaled, dtype=torch.float32)).numpy()*saved["y_std"] + saved["y_mean"]
            say(f"fit_{name}", {g: {"share_pct": round(100*float(m.mean()), 1), "r2": round(r2(pred[m], new_v["y"][m]), 3),
                                    "rmse": round(rmse(pred[m], new_v["y"][m]), 2)} for g, m in groups.items()})

    say("A_old_data_2x64", fit(old_x, old_y, old_v["x"][:, NO_VEL], old_v["y"], 64, 2))
    say("B_rooms_2x64", fit(x, new["y"], xv, new_v["y"], 64, 2))
    say("C_rooms_3scans_2x64", fit(x3, new["y"], xv3, new_v["y"], 64, 2))
    say("D_rooms_3x256", fit(x, new["y"], xv, new_v["y"], 256, 3))
    say("E_rooms_true_mover_state_2x64", fit(np.hstack([x, new["mover"]]), new["y"], np.hstack([xv, new_v["mover"]]), new_v["y"], 64, 2))
    say("F_rooms_true_mover_state_3x256", fit(np.hstack([x, new["mover"]]), new["y"], np.hstack([xv, new_v["mover"]]), new_v["y"], 256, 3))
    say("seconds", round(time.time() - t))
