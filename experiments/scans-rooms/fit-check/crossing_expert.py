"""What did the window expert want before each crossing hit? (4 Nov.) Uses the shadow-labelled reruns in
fit-check/crossing-labels (same model, same rooms as r3-validation). For the 4 s before the first mover contact:
model vs expert forward speed and turn rate, the expert's evade flag, and when they first disagreed."""
import json
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path("/home/akhjd/ros2_ws/comp0245_project/runs/scans-rooms/fit-check")
R, TRACK, DT = 0.11, 0.66, 0.01


def body(wheels):
    """Wheel speeds (FR, FL, BL, BR) -> forward speed, turn rate (skid steer)."""
    left, right = (wheels[:, 1] + wheels[:, 2])/2, (wheels[:, 0] + wheels[:, 3])/2
    return R*(left + right)/2, R*(right - left)/TRACK


if __name__ == "__main__":
    folder = HERE/"crossing-labels"
    val = Path("/home/akhjd/ros2_ws/comp0245_project/runs/dagger-rooms-window72b/r3-validation")
    crossing = {r["id"] for r in json.loads((val.parent/"r3-validation-crossing-detail.json").read_text())}
    rows = []
    for e in json.loads((folder/"manifest.json").read_text())["episodes"]:
        z = np.load(folder/e["path"]/"trajectory.npz")
        mv = np.array(e["obstacle_is_moving"], bool)
        hit = z["contact_obstacle"][:, mv].astype(bool)
        zv = np.load(val/e["path"]/"trajectory.npz")
        n = min(len(z["base_m"]), len(zv["base_m"]))
        if not hit.any():
            rows.append({"id": e["path"], "note": "no mover hit in the rerun"})
            continue
        t0 = int(np.argwhere(hit)[0][0])
        same = float(np.abs(z["base_m"][:t0, :2] - zv["base_m"][:t0, :2]).max()) if t0 <= n else None
        vm, wm = body(z["actions"][:, :4])
        ve, we = body(z["expert_wheels"])
        ev = z["expert_evade"].astype(bool) if "expert_evade" in z.files else np.zeros(len(vm), bool)
        a = max(t0 - 400, 0)
        dv, dw = np.abs(vm[a:t0] - ve[a:t0]), np.abs(wm[a:t0] - we[a:t0])
        apart = np.flatnonzero((dv > 0.08) | (dw > 0.25))
        out = {"id": e["path"], "crossing": e["path"] in crossing, "rerun_matches_m": round(same, 3) if same is not None else None,
               "disagree_from_s_before": round((t0 - a - apart[0])*DT, 1) if len(apart) else 0.0,
               "disagree_share_4s": round(float(len(apart)/max(t0 - a, 1)), 2),
               "evade_share_4s": round(float(ev[a:t0].mean()), 2)}
        for s in (3, 2, 1, 0.5):
            i = max(t0 - int(s/DT), 0)
            sl = slice(max(i - 10, 0), i + 10)
            out[f"{s}s"] = {"model_v": round(float(vm[sl].mean()), 2), "expert_v": round(float(ve[sl].mean()), 2),
                            "model_w": round(float(wm[sl].mean()), 2), "expert_w": round(float(we[sl].mean()), 2)}
        e2 = slice(max(t0 - 200, 0), t0)
        out["expert_wants_2s"] = ("faster forward" if ve[e2].mean() > vm[e2].mean() + 0.05 else
                                  "back up" if ve[e2].mean() < vm[e2].mean() - 0.05 else
                                  "turn" if np.abs(we[e2] - wm[e2]).mean() > 0.2 else "about the same")
        rows.append(out)
        print(out, flush=True)
    c = [r for r in rows if r.get("crossing")]
    print(len(rows), "reruns,", len(c), "crossing hits;", sum("note" in r for r in rows), "reruns without a mover hit")
    print("rerun path matches the scored run (max deviation before contact, m):", [r["rerun_matches_m"] for r in c])
    print("expert wanted, last 2 s (crossing):", Counter(r["expert_wants_2s"] for r in c))
    print("disagreement started (s before contact):", sorted(r["disagree_from_s_before"] for r in c))
    print("expert in evade mode, share of last 4 s, median:", np.median([r["evade_share_4s"] for r in c]))
    (HERE/"crossing_expert.json").write_text(json.dumps(rows, indent=1))
