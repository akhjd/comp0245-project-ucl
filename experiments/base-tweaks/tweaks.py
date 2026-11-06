"""Two small changes to the room base, tested base-only on the 60 validation rooms (arm folded, 70 s), as
runs/crossing/eval_fold.py (5 Nov, AGENT_LOG 13:10). eval_obstacles.LearnedBase itself is unchanged.
    ensemble   seed 0 and seed 1 crossing r3 + latched hold: wheel commands averaged, hold probabilities averaged,
               then the usual latched hold on the average
    turnhold   seed 0 model; while it holds, forward/backward speed is 0 but the network's turn is kept
Usage: tweaks.py VARIANT OUT_FOLDER [WORKERS]"""
import contextlib
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path
import numpy as np

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")    # runs/ is a symlink to /data, so not from __file__
sys.path.insert(0, str(ROOT))
from eval_obstacles import LearnedBase

SEED0 = ROOT/"runs/crossing-hold/r3-hold-latch/model.pt"
SEED1 = ROOT/"runs/crossing-hold/s1-r3-hold-latch/model.pt"


class NoHold(LearnedBase):
    """LearnedBase that logs its hold probability but never holds (the caller decides)."""
    def holding(self, p_hold):
        return False


class EnsembleBase:
    """Two networks with the same inputs; drives with the mean of their wheel commands and holds on the mean hold
    probability with LearnedBase's latched rule."""
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.hold_latch, self.hold_run, self.latched = True, 0, False
        self.hold_log = []

    def act(self, state):
        out_a, out_b = self.a.act(state), self.b.act(state)
        p = (self.a.hold_log[-1] + self.b.hold_log[-1])/2
        self.hold_log.append(p)
        wheels = (out_a[:4] + out_b[:4])/2
        if LearnedBase.holding(self, p):
            wheels = np.zeros(4)
        return np.concatenate([wheels, out_a[4:]])     # arm: the same PD hold in both

    def save_modes(self, folder):
        np.savez_compressed(Path(folder)/"hold.npz", hold_p=np.array(self.hold_log, np.float32))


class TurnHold(LearnedBase):
    """While holding: no forward/backward speed, but the network's own turn (wheels [FR, FL, BL, BR])."""
    def holding(self, p_hold):
        self.hold_now = super().holding(p_hold)
        return False

    def act(self, state):
        out = super().act(state)
        if self.hold_now:
            w = out[:4]
            turn = ((w[0] + w[3]) - (w[1] + w[2]))/4      # half the right-minus-left difference: pure spin
            out = np.concatenate([[turn, -turn, -turn, turn], out[4:]])
        return out


def policy(variant):
    from eval_obstacles import load_model
    if variant == "ensemble":
        return EnsembleBase(NoHold(*load_model(SEED0)), NoHold(*load_model(SEED1)))
    if variant == "turnhold":
        return TurnHold(*load_model(SEED0))
    raise ValueError(variant)


def episode(job):
    variant, scenario, folder = job
    import torch
    torch.set_num_threads(1)
    from eval_obstacles import score
    from obstacle_env import rollout
    from pipeline import read_json
    config = read_json(ROOT/"settings.json")
    config["duration_s"], config["lidar_rays"] = 70.0, 72
    p = policy(variant)
    with open(os.devnull, "w") as quiet, contextlib.redirect_stderr(quiet):
        summary = rollout(config, scenario, folder, p, save_every=10)
    p.save_modes(folder)
    return summary, score(Path(folder), scenario, summary)


if __name__ == "__main__":
    from eval_obstacles import ARM_FOLD
    from compare_scans import contacts
    variant, out = sys.argv[1], ROOT/sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    if out.exists():
        raise FileExistsError(f"Choose a new output folder: {out}")
    out.mkdir(parents=True)
    rooms = [{**s, "arm_q0": ARM_FOLD} for f in ("scenarios_rooms_train.json", "scenarios_rooms_val.json")
             for s in json.loads((ROOT/f).read_text()) if s["split"] == "validation"]
    with get_context("spawn").Pool(workers) as pool:
        results = pool.map(episode, [(variant, s, str(out/s["id"])) for s in rooms], chunksize=1)
    (out/"manifest.json").write_text(json.dumps({"saved_every": 10, "lidar_rays": 72, "episodes": [
        {"path": s["id"], "scenario_id": s["id"], "split": s["split"], **summary} for s, (summary, _) in zip(rooms, results)]}, indent=1))
    scores = [r for _, r in results]
    totals = {"episodes": len(scores), "parked": sum(r["parked"] for r in scores),
              "parked_worst_case_reach": sum(r["parked_worst_case_reach"] for r in scores), **contacts(out)}
    (out/"evaluation.json").write_text(json.dumps({"variant": variant, "totals": totals, "episodes": scores}, indent=1))
    print(variant, totals, flush=True)
