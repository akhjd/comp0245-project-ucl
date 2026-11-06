"""Output check (4 Nov): which output stops the room model hedging near movers? Same network as the window DAgger
(ray filter + threat, 3 scans, no velocity, 72 rays), plain cloning on rooms-window72, scored on rooms-window72-val:
regression, mixture 3, regression + evade weight 3, menu, menu + evade weight 3 (evade = the teacher's own flag).
Per variant on validation rows: turn and speed correlation with the expert and mean |turn| model / expert, near movers
(< 1 m) and on all rows; all-rows wheel R2.
Rule (AGENT_LOG 11:50, written before the results): passes if near-mover turn correlation beats regression by >= 0.1
AND |turn| ratio >= 0.75, with all-rows wheel R2 at most 0.05 below regression. Usage: output_check.py [--smoke]"""
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")    # runs/ is a symlink to /data, so not from __file__
HERE = ROOT/"runs/scans-rooms/fit-check"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from train_base_torch import train, load, speed_turn
from eval_obstacles import load_model
from size_check import near_movers, r2

VARIANTS = {"regression": {}, "mixture3": {"mixture": 3}, "regression_evade3": {"evade_weight": 3.0},
            "menu": {"menu": True}, "menu_evade3": {"menu": True, "evade_weight": 3.0}}


def corr(a, b):
    return round(float(np.corrcoef(a, b)[0, 1]), 3)


def crossing_rows():
    """Inputs and expert labels for the last 4 s before each mover hit in the shadow-labelled reruns of window r3
    (fit-check/crossing-labels: validation rooms, analysis only). Extra information, not part of the rule."""
    folder = HERE/"crossing-labels"
    x, y = load("analysis", folder, "expert_wheels", scans=3)[:2]
    keep = []
    for e in json.loads((folder/"manifest.json").read_text())["episodes"]:
        z = np.load(folder/e["path"]/"trajectory.npz")
        mv = np.array(e["obstacle_is_moving"], bool)
        hit = z["contact_obstacle"][:, mv].astype(bool).any(axis=1) if mv.any() else np.zeros(len(z["time_s"]), bool)
        k = np.zeros(len(z["time_s"]), bool)
        if hit.any():
            t0 = int(np.flatnonzero(hit)[0])
            k[max(t0 - 400, 0):t0] = True
        keep.append(k)
    keep = np.concatenate(keep)
    return x[keep], y[keep]


def score(model_path, x, y, near, evade, crossing=None):
    model, saved = load_model(model_path)
    xs = ((x[:, saved["columns"]] - saved["x_mean"])/saved["x_std"]).astype(np.float32)
    with torch.no_grad():
        pred = np.concatenate([model(torch.from_numpy(xs[i:i + 65536])).numpy() for i in range(0, len(xs), 65536)])
    pred = pred*saved["y_std"] + saved["y_mean"]
    (vm, wm), (ve, we) = speed_turn(pred), speed_turn(y)
    out = {"wheel_r2_all": round(r2(pred, y), 3)}
    for name, m in (("near", near), ("evade", evade), ("all", np.ones(len(y), bool))):
        out[name] = {"turn_corr": corr(wm[m], we[m]), "speed_corr": corr(vm[m], ve[m]),
                     "turn_ratio": round(float(np.abs(wm[m]).mean()/np.abs(we[m]).mean()), 3)}
    if crossing is not None:
        out["before_hits"] = score(model_path, *crossing, np.ones(len(crossing[1]), bool), np.ones(len(crossing[1]), bool))["near"]
    return out


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    torch.set_num_threads(8)
    data = ROOT/("runs/teacher-compare/window" if smoke else "runs/rooms-window72")
    val = ROOT/("runs/teacher-compare/window-val" if smoke else "runs/rooms-window72-val")
    out_dir = HERE/("output-check-smoke" if smoke else "output-check")
    result_file = HERE/("output_check_smoke.json" if smoke else "output_check.json")
    if smoke:
        shutil.rmtree(out_dir, ignore_errors=True)
        result_file.unlink(missing_ok=True)
    xv, yv, *_, ev = load("validation", val, scans=3, mode_label="expert")
    near = near_movers(val, "validation")
    crossing = None if smoke else crossing_rows()
    results = json.loads(result_file.read_text()) if result_file.exists() else {}
    for name, extra in VARIANTS.items():
        if name in results:
            continue
        train([str(data)], [], out_dir/name, True, epochs=1 if smoke else 30, seed=0, scans=3, encoder="conv",
              threat=True, mode_label="expert", validation=str(val), **extra)
        results[name] = score(out_dir/name/"model.pt", xv, yv, near, ev, crossing)
        print(name, results[name], flush=True)
        result_file.write_text(json.dumps(results, indent=1))
    base = results["regression"]
    for name, r in results.items():
        r["passes"] = bool(r["near"]["turn_corr"] >= base["near"]["turn_corr"] + 0.1 and r["near"]["turn_ratio"] >= 0.75
                           and r["wheel_r2_all"] >= base["wheel_r2_all"] - 0.05) if name != "regression" else None
        print(name, "near", r["near"], "R2", r["wheel_r2_all"], "passes", r["passes"])
    result_file.write_text(json.dumps(results, indent=1))
