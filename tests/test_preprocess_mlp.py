"""Tests for the MLP on preprocessed lidar (comp0245/training/train_mlp.py --preprocess,
comp0245/training/dagger_mlp.py, comp0245/training/build_features.py).

    python -m unittest tests.test_preprocess_mlp        run from the project folder; needs \\
        runs/dataset

The key check: the driving path (LearnedWheels, feeding the Preprocessor scan by scan) gives the
same wheel commands as the training path (cached features) on a real saved run.
"""

import argparse
import glob
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from comp0245 import preprocess as pp
from comp0245.training import train_mlp as t

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "runs/dataset"


def settings(**changes):
    base = dict(
        no_velocity=True,
        no_xy=True,
        dot_polar=True,
        lidar_features=False,
        preprocess=True,
        seed=0,
        mirror=False,
        hidden=32,
        layers=2,
        lr=3e-4,
        lr_patience=4,
        resume=False,
        epochs=1,
        batch=1024,
        patience=50,
        lidar_noise=0.0,
        velocity_noise=0.0,
        yaw_rate_noise=0.0,
        dropout=0.2,
        weight_decay=0.01,
        evals_per_epoch=1,
    )
    return argparse.Namespace(**{**base, **changes})


@unittest.skipUnless(DATA.exists(), "needs runs/dataset")
class PreprocessedMLPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        # Copies of a few real runs, so the test's caches never touch the dataset
        cls.tmp = Path(tempfile.mkdtemp())
        for split, n in (("train", 3), ("validation", 2)):
            found = 0
            for name in sorted(glob.glob(str(DATA / "case4/part*/*/trajectory.npz"))):
                run = Path(name).parent
                scenario = json.loads((run / "episode.json").read_text())["scenario"]
                summary = json.loads((run / "summary.json").read_text())
                if scenario["split"] == split and summary["success"] and found < n:
                    target = cls.tmp / "data/case4/part0" / run.name
                    target.mkdir(parents=True)
                    for f in ("trajectory.npz", "summary.json", "episode.json"):
                        shutil.copy(run / f, target / f)
                    found += 1
        cls.train = t.load(str(cls.tmp / "data"), "train", preprocess_=True)
        cls.val = t.load(str(cls.tmp / "data"), "validation", preprocess_=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_cache_is_written_and_reused(self):
        caches = list((self.tmp / "data").glob(f"*/*/*/features_{pp.VERSION}.npy"))
        self.assertEqual(len(caches), 5)
        self.assertEqual(self.train.pre.shape, (len(self.train.wheels), len(pp.MLP_NAMES)))
        self.assertEqual(self.train.lidar.shape[1], 0)  # raw lidar not kept
        again = t.load(str(self.tmp / "data"), "train", preprocess_=True)
        self.assertTrue(np.array_equal(again.pre, self.train.pre))

    def test_trains_with_the_chosen_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            t.fit(self.train, self.val, Path(tmp), settings())
            _, saved = t.load_policy(Path(tmp) / "best.pt")
        names = saved["input_names"]
        self.assertEqual(
            len(names), len(pp.MLP_NAMES) + 2 + 15
        )  # features, sin and cos yaw, polar dots
        self.assertFalse({"x", "y", "v_forward", "v_left", "yaw_rate"} & set(names))
        self.assertFalse(any(n.startswith(("mover_", "object_", "lidar_t-")) for n in names))
        self.assertTrue(saved["input_changes"]["preprocess"])

    def test_door_memory_trains_and_drives_the_same(self):
        """Feature set v2 (--door-memory): its own cache, its inputs, and driving equal to
        training on a real run."""
        train = t.load(str(self.tmp / "data"), "train", preprocess_=True, version="v2")
        val = t.load(str(self.tmp / "data"), "validation", preprocess_=True, version="v2")
        self.assertEqual(len(list((self.tmp / "data").glob("*/*/*/features_v2.npy"))), 5)
        self.assertTrue(
            np.array_equal(train.pre[:, : len(pp.MLP_NAMES)], self.train.pre)
        )  # v1 part unchanged
        with tempfile.TemporaryDirectory() as tmp:
            t.fit(train, val, Path(tmp), settings(door_memory=True))
            model, saved = t.load_policy(Path(tmp) / "best.pt")
        self.assertEqual(len(saved["input_names"]), len(pp.FEATURE_SETS["v2"]) + 2 + 15)
        self.assertTrue(saved["input_changes"]["door_memory"])
        self.check_driving(model, saved, "v2")

    def test_driving_matches_training_on_a_real_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            t.fit(self.train, self.val, Path(tmp), settings())
            model, saved = t.load_policy(Path(tmp) / "best.pt")
        self.check_driving(model, saved, "v1")

    def check_driving(self, model, saved, version):
        from comp0245.training.dagger_mlp import CONTROL_EVERY, LearnedWheels

        scale = t.Scale(saved=saved)
        run = sorted((self.tmp / "data").glob("*/*/*/trajectory.npz"))[0].parent
        z = np.load(run / "trajectory.npz")
        summary = json.loads((run / "summary.json").read_text())
        scenario = json.loads((run / "episode.json").read_text())["scenario"]
        rows = 400
        wheels = LearnedWheels(model, saved)
        robot = type(
            "Robot",
            (),
            {
                "dots": [
                    np.asarray(d, float) for d in scenario.get("targets", [scenario["target"]])
                ],
                "picked": [p for p in summary["picks"]],
            },
        )()
        driven = []
        for i in range(rows):
            robot.picked = [p for p in summary["picks"] if p["time_s"] <= z["time_s"][i]]
            state = {"lidar": z["lidar"][i], "x": z["observations"][i]}
            driven.append(wheels(state, robot, z["time_s"][i]).copy())
            for _ in range(CONTROL_EVERY - 1):  # the 1 ms steps between commands
                wheels(state, robot, z["time_s"][i])
        data = t.load(str(self.tmp / "data"), scenario["split"], preprocess_=True, version=version)
        first = int(
            np.flatnonzero(np.r_[True, data.start[1:] != data.start[:-1]])[
                sorted(
                    p.parent.name
                    for p in (self.tmp / "data").glob("*/*/*/trajectory.npz")
                    if json.loads((p.parent / "episode.json").read_text())["scenario"]["split"]
                    == scenario["split"]
                ).index(run.name)
            ]
        )
        trained = t.predict(model, scale, data, np.arange(first, first + rows))
        self.assertLess(np.abs(np.array(driven) - trained).max(), 1e-4)


if __name__ == "__main__":
    unittest.main()
