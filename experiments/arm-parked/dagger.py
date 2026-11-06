"""Arm DAgger on parked spots (5 Nov, AGENT_LOG 12:05). Each round: the previous arm reaches from fresh parked spots
(make_scenarios.py, base parked, arm from its normal pose) while a shadow copy of the arm expert labels every step
(obstacle_env.rollout label_arm, as dagger_arm.py); then train_arm.py retrains on the expert data, the old DAgger
folders and every round so far. Then the arm-only tests: the 49 validation spots and the synthetic held-out spots.
Usage: dagger.py [ROUNDS]"""
import contextlib
import json
import os
import subprocess
import sys
from multiprocessing import get_context
from pathlib import Path
import numpy as np

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")    # runs/ is a symlink to /data, so not from __file__
sys.path.insert(0, str(ROOT))
HERE = ROOT/"runs/arm-parked"
PY = "/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python"
DATA = ["runs/obst-v4-height", "runs/obst-v4-armstart", "runs/obst-v4-both", "runs/arm-parked/data"]
OLD_DAGGER = json.loads((ROOT/"runs/dagger-arm-b/r3/training.json").read_text())["dagger"]
SPOTS, DURATION, EPOCHS, WORKERS, SAVE_EVERY = 500, 15.0, 60, 12, 10


class Parked:
    """Stands in for the base: already parked, wheels at 0."""
    arrived = True

    def act(self, state):
        return np.zeros(11)


def episode(job):
    arm_path, scenario, folder = job
    import torch
    torch.set_num_threads(1)
    from obstacle_env import rollout
    from pipeline import read_json
    from eval_arm import LearnedArm, load_arm
    config = read_json(ROOT/"settings.json")
    config["duration_s"] = DURATION
    policy = LearnedArm(*load_arm(arm_path), scenario["target"], Parked())
    with open(os.devnull, "w") as quiet, contextlib.redirect_stderr(quiet):
        return rollout(config, scenario, folder, policy, label_arm=True, save_every=SAVE_EVERY)


def run(cmd, log):
    with open(log, "w") as f:
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    arm, folders = "runs/arm-parked/arm/model.pt", []
    for k in range(1, rounds + 1):
        out = HERE/f"dagger-r{k}"
        out.mkdir()
        run([PY, "-B", "runs/arm-parked/make_scenarios.py", str(out/"spots.json"), str(SPOTS), str(20261106 + k)], out/"spots.log")
        spots = json.loads((out/"spots.json").read_text())
        data = out/"data"
        data.mkdir()
        with get_context("spawn").Pool(WORKERS) as pool:
            summaries = pool.map(episode, [(str(ROOT/arm), s, str(data/s["id"])) for s in spots], chunksize=1)
        (data/"manifest.json").write_text(json.dumps({"saved_every": SAVE_EVERY, "episodes": [
            {"path": s["id"], "scenario_id": s["id"], "split": "train", **m} for s, m in zip(spots, summaries)]}, indent=1))
        print(f"round {k}: arm {arm} reached {sum(m['reached_target'] for m in summaries)}/{len(spots)} while driving", flush=True)
        folders.append(f"runs/arm-parked/dagger-r{k}/data")
        run([PY, "-B", "train_arm.py", "--data", *DATA, "--dagger", *OLD_DAGGER, *folders, "--epochs", str(EPOCHS),
             "--seed", "0", "--out", f"runs/arm-parked/dagger-r{k}/arm"], out/"train.log")
        arm = f"runs/arm-parked/dagger-r{k}/arm/model.pt"
        run([PY, "-B", "runs/arm-parked/arm_test.py", arm, f"runs/arm-parked/dagger-r{k}/test-val49", str(WORKERS)], out/"test-val49.log")
        run([PY, "-B", "runs/arm-parked/arm_test.py", arm, f"runs/arm-parked/dagger-r{k}/test-synth200", str(WORKERS),
             "runs/arm-parked/synth200.json"], out/"test-synth200.log")
        for t in ("test-val49", "test-synth200"):
            r = json.loads((out/t/"results.json").read_text())
            print(f"round {k} {t}: success {r['success']}/{r['spots']}", flush=True)
    print("ARM DAGGER FINISHED", flush=True)
