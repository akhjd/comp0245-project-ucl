"""Why does the switch miss? (3 Nov, side session lidar72-modes; offline, saved expert data only, no test data)

1. Visibility: on the validation evade rows, is the mover that makes the row "evade" hit by any lidar ray now
   (obstacle_seen), or in the last 1 s (the span of the 3 scans)?
2. Switch alone: the same RayNet trained ONLY on the switch's cross-entropy (no wheel loss, so early stopping is not
   decided by the wheels), 10 epochs, caught / false alarm at 0.5 per epoch on the validation rows. Once without the
   3 velocity inputs (as the gate model) and once with them (the label depends on the robot's own velocity).
Writes switch_diag.json next to this file.
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).absolute().parents[2]   # not resolve(): runs/ is a symlink to /data
sys.path.insert(0, str(ROOT))
from train_base_torch import (load, input_names, build_net, collision_labels, VELOCITY, BATCH, LR,   # noqa: E402
                              MODE_HORIZON_S, MODE_STEP_S, MODE_MARGIN_M)
from obstacle_env import ROBOT_RADIUS                                                                 # noqa: E402

HERE = Path(__file__).absolute().parent
TRAIN, VALIDATION = ROOT/"runs"/"rooms-expert", ROOT/"runs"/"rooms-expert-val"


def visibility():
    """Share of evade rows whose threatening mover(s) were hit by a ray now / within the last 1 s."""
    manifest = json.loads((VALIDATION/"manifest.json").read_text())
    now = recent = total = 0
    for ep in manifest["episodes"]:
        if ep["split"] != "validation" or not ep["success"]:
            continue
        z = np.load(VALIDATION/ep["path"]/"trajectory.npz")
        if "obstacle_xy" not in z.files:
            continue
        moving = np.asarray(ep["obstacle_is_moving"], bool)
        radii = [o["r"] for o in json.loads((VALIDATION/ep["path"]/"episode.json").read_text())["scenario"]["obstacles"]]
        evade = collision_labels(z, moving, radii, 10)
        # Per mover: would this mover alone make the row evade? (label per mover)
        per = np.stack([collision_labels(z, np.arange(len(moving)) == k, radii, 10) for k in np.where(moving)[0]], axis=1)
        seen = z["obstacle_seen"][::10][:, moving]
        window = int(round(1.0/0.01))
        seen_recent = np.array([seen[max(0, i - window):i + 1].any(axis=0) for i in range(len(seen))])
        rows = np.where(evade)[0]
        total += len(rows)
        now += int((per[rows] & seen[rows]).any(axis=1).sum())
        recent += int((per[rows] & seen_recent[rows]).any(axis=1).sum())
    return {"evade_rows": total, "threat_seen_now_pct": 100*now/total, "threat_seen_last_1s_pct": 100*recent/total}


def switch_only(no_velocity, epochs=10, seed=0):
    torch.manual_seed(seed)
    x, _, _, m = load("train", TRAIN, scans=3, mode_label="collision")
    xv, _, _, mv = load("validation", VALIDATION, scans=3, mode_label="collision")
    names = input_names(3)
    cols = [i for i, n in enumerate(names) if not (no_velocity and n in VELOCITY)]
    x, xv = x[:, cols], xv[:, cols]
    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-6
    lid = slice(x.shape[1] - 108, x.shape[1])
    mean[lid], std[lid] = x[:, lid].mean(), x[:, lid].std() + 1e-6
    X, XV = torch.tensor((x - mean)/std), torch.tensor((xv - mean)/std)
    M, MV = torch.tensor(m.astype(np.float32)), torch.tensor(mv.astype(np.float32))
    net = build_net(x.shape[1], 64, 3, "conv", modes=True)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    bce = nn.functional.binary_cross_entropy_with_logits
    table = []
    for epoch in range(epochs):
        net.train()
        order = torch.randperm(len(X))
        for start in range(0, len(X), BATCH):
            idx = order[start:start + BATCH]
            loss = bce(net.heads(net.features(X[idx]))[0], M[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            logit = net.heads(net.features(XV))[0]
            val = bce(logit, MV).item()
        alarm, real = (logit > 0).numpy(), mv.astype(bool)
        table.append({"epoch": epoch + 1, "validation_bce": val, "caught_pct": 100*float(alarm[real].mean()),
                      "false_alarm_pct": 100*float((~real[alarm]).mean()) if alarm.any() else float("nan")})
        print(no_velocity, table[-1], flush=True)
    return table


def main():
    out = {"label": {"horizon_s": MODE_HORIZON_S, "step_s": MODE_STEP_S, "margin_m": MODE_MARGIN_M,
                     "robot_radius_m": ROBOT_RADIUS},
           "visibility": visibility()}
    print(out, flush=True)
    out["switch_only_no_velocity"] = switch_only(True)
    out["switch_only_with_velocity"] = switch_only(False)
    (HERE/"switch_diag.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
