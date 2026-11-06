"""Offline gate for the evade/target switch (3 Nov, side session lidar72-modes; AGENT_LOG 22:26 PLAN).

On the room validation rows (runs/rooms-expert-val, collision labels): share of evade rows; for the switch at
thresholds 0.4 / 0.5 / 0.6, % of real evade rows it catches (recall) and % of its alarms that are false (1 - precision);
the same with the driving hysteresis (enter 0.6, leave 0.4) run along each validation run; wheel RMSE (rad/s) of each
head on its own rows (true label picks the head) and of the switch-picked output, against the plain conv model.
Writes gate.json next to this file. Akhilesh's bar: over 75% caught and under 30% false alarms.
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).absolute().parents[2]   # not resolve(): runs/ is a symlink to /data
sys.path.insert(0, str(ROOT))
from eval_obstacles import load_model, SWITCH_ON, SWITCH_OFF   # noqa: E402
from train_base_torch import load                              # noqa: E402

HERE = Path(__file__).resolve().parent
VALIDATION = ROOT/"runs"/"rooms-expert-val"


def scaled(model_path, x):
    model, saved = load_model(model_path)
    xs = torch.tensor(((x[:, saved["columns"]] - saved["x_mean"])/saved["x_std"]).astype(np.float32))
    return model, saved, xs


def rates(alarm, real):
    caught = float(alarm[real].mean()) if real.any() else float("nan")
    false = float((~real[alarm]).mean()) if alarm.any() else float("nan")
    return {"caught_pct": 100*caught, "false_alarm_pct": 100*false, "alarm_rows_pct": 100*float(alarm.mean())}


def rmse(pred, y, rows):
    return float(np.sqrt(np.mean((pred[rows] - y[rows])**2)))


def main():
    manifest = json.loads((VALIDATION/"manifest.json").read_text())
    runs = [ep for ep in manifest["episodes"] if ep["split"] == "validation" and ep["success"]]
    x, y, _, real = load("validation", VALIDATION, scans=3, mode_label="collision")
    real = real.astype(bool)
    out = {"validation": str(VALIDATION), "runs": len(runs), "rows": len(x), "evade_share_pct": 100*float(real.mean())}

    model, saved, xs = scaled(HERE/"modes"/"model.pt", x)
    with torch.no_grad():
        logit, both = model.heads(model.features(xs))
    p = torch.sigmoid(logit).numpy()
    both = both.numpy()*saved["y_std"] + saved["y_mean"]
    out["thresholds"] = {str(t): rates(p > t, real) for t in (0.4, 0.5, 0.6)}

    # Hysteresis along each run (rows are 10 ms apart here; LearnedBase applies it every 1 ms)
    lengths, start = [], 0
    for ep in runs:   # load() keeps runs in manifest order; lengths from the same stride
        n = len(np.load(VALIDATION/ep["path"]/"trajectory.npz")["observations"][::10])
        lengths.append((start, start + n))
        start += n
    assert start == len(x), (start, len(x))
    held = np.zeros(len(x), bool)
    for a, b in lengths:
        evading = False
        for i in range(a, b):
            evading = (p[i] >= SWITCH_OFF) if evading else (p[i] > SWITCH_ON)
            held[i] = evading
    out["hysteresis"] = {**rates(held, real), "head_changes_per_run": float(np.mean(
        [np.count_nonzero(np.diff(held[a:b])) for a, b in lengths]))}

    target_rows, evade_rows = ~real, real
    own = np.where(real[:, None], both[:, 1], both[:, 0])
    picked = np.where(held[:, None], both[:, 1], both[:, 0])
    out["modes_rmse"] = {"target_head_on_target_rows": rmse(both[:, 0], y, target_rows),
                         "evade_head_on_evade_rows": rmse(both[:, 1], y, evade_rows),
                         "true_label_heads_all_rows": rmse(own, y, slice(None)),
                         "switch_picked_all_rows": rmse(picked, y, slice(None)),
                         "switch_picked_target_rows": rmse(picked, y, target_rows),
                         "switch_picked_evade_rows": rmse(picked, y, evade_rows)}

    plain, psaved, pxs = scaled(HERE/"plain-conv"/"model.pt", x)
    with torch.no_grad():
        pred = plain(pxs).numpy()*psaved["y_std"] + psaved["y_mean"]
    out["plain_conv_rmse"] = {"target_rows": rmse(pred, y, target_rows), "evade_rows": rmse(pred, y, evade_rows),
                              "all_rows": rmse(pred, y, slice(None))}
    for name in ("modes", "plain-conv"):
        info = json.loads((HERE/name/"training.json").read_text())
        out[f"{name}_training"] = {k: info[k] for k in ("best_epoch", "epochs", "parameters",
                                                         "validation_wheel_rmse_rad_s")}
    bar = out["thresholds"]["0.5"]
    out["bar_passed_at_0.5"] = bool(bar["caught_pct"] > 75 and bar["false_alarm_pct"] < 30)
    (HERE/"gate.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
