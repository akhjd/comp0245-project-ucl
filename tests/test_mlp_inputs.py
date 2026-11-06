"""Tests for the wheels MLP's input options (comp0245/training/train_mlp.py,
comp0245/training/dagger_mlp.py): --no-xy, --dot-polar, --lidar-features, --run-fraction.

    python -m unittest tests.test_mlp_inputs        run from the project folder; needs \\
        runs/dataset and runs/mlp3

Uses one thread and a few runs, so it can run beside a long job. Takes a few minutes, mostly the
short simulator drives.
"""

import argparse
import glob
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from comp0245.training import train_mlp as t

ROOT = Path(__file__).resolve().parent.parent
DATA = str(ROOT / "runs/dataset")
HAVE_DATA = Path(DATA).exists()


def hand_made_row():
    """Obstacle 1 m away at ray 9 (90 degrees, left), 2 m there 0.5 s ago; dot 0 3 m straight to
    the left; dots 1 and 2 absent."""
    x = np.full((1, 199), 4.0, np.float32)
    x[0, 9] = 1.0
    x[0, 72 + 9] = 2.0
    x[0, 180:199] = 0
    x[0, 187:191] = [0, 3, 0.5, 1]
    return x


class FeatureTests(unittest.TestCase):
    def test_features_give_the_right_values(self):
        x = hand_made_row()
        f = t.make_features(x, True, True)[0]
        names = t.feature_names(True, True)
        self.assertEqual(len(names), len(f))
        v = dict(zip(names, f, strict=False))
        self.assertEqual(
            (v["sector90deg_nearest"], v["sector90deg_change"], v["sector0deg_nearest"]),
            (1, -1, 4),
        )
        self.assertEqual(v["closest_distance"], 1)
        self.assertAlmostEqual(v["closest_sin"], 1, places=5)
        self.assertAlmostEqual(v["closest_cos"], 0, places=5)
        self.assertEqual((v["dot0_free"], v["dot1_free"]), (1, 4))  # an absent dot reads as clear
        self.assertAlmostEqual(v["dot0_distance"], 3, places=5)
        self.assertAlmostEqual(v["dot0_sin_heading"], 1, places=5)
        self.assertAlmostEqual(v["dot0_cos_heading"], 0, places=5)
        self.assertEqual((v["dot0_up"], v["dot0_there"]), (0.5, 1))
        self.assertEqual((v["dot1_distance"], v["dot1_cos_heading"], v["dot1_there"]), (0, 0, 0))

    def test_no_changes_leave_raw_inputs(self):
        x = hand_made_row()
        self.assertTrue((t.make_features(x) == x).all())

    def test_input_counts(self):
        self.assertEqual(len(t.kept_columns()), 199)
        self.assertEqual(len(t.kept_columns(no_velocity=True)), 196)
        self.assertEqual(len(t.kept_columns(no_xy=True)), 197)
        self.assertEqual(len(t.kept_columns(dot_polar=True)), 202)
        self.assertEqual(len(t.kept_columns(lidar_features=True)), 49)


@unittest.skipUnless(HAVE_DATA, "needs runs/dataset")
class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.train = t.load(DATA, "train", limit=8)
        cls.val = t.load(DATA, "validation", limit=4)

    def test_run_fraction_keeps_nested_subsets(self):
        sizes = [
            t.load(DATA, "train", cases=("case1",), run_fraction=f).runs for f in (1.0, 0.5, 0.25)
        ]
        self.assertTrue(sizes[0] > sizes[1] > sizes[2] > 0, sizes)
        self.assertEqual(
            t.load(DATA, "validation", cases=("case1",)).runs,
            t.load(DATA, "validation", cases=("case1",), run_fraction=1.0).runs,
        )

    @unittest.skipUnless((ROOT / "runs/mlp3/best.pt").exists(), "needs runs/mlp3")
    def test_old_model_inputs_unchanged(self):
        _, saved = t.load_policy(ROOT / "runs/mlp3/best.pt")
        scale, val = t.Scale(saved=saved), self.val
        idx = np.arange(0, len(val.wheels), 50)
        past = np.maximum(idx[:, None] - t.OFFSETS, val.start[idx, None])
        raw = np.hstack([val.lidar[past].reshape(len(idx), -1), val.state[idx]])
        self.assertTrue(
            np.array_equal(
                scale.inputs(val, idx).numpy(), ((raw - scale.x_mean) / scale.x_std)[:, scale.keep]
            )
        )

    def test_each_option_trains_matches_driving_and_drives(self):
        from comp0245.training import dagger_mlp as d

        scenario = json.loads(
            (
                Path(sorted(glob.glob(f"{DATA}/case1/part*/*/trajectory.npz"))[0]).parent
                / "episode.json"
            ).read_text()
        )["scenario"]
        options = {
            "no_xy": (["--no-velocity", "--no-xy"], 194),
            "polar": (["--no-velocity", "--dot-polar"], 199),
            "lidar": (["--no-velocity", "--lidar-features"], 46),
        }
        for name, (flags, size) in options.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "m"
                out.mkdir()
                args = argparse.Namespace(
                    no_velocity=True,
                    no_xy="--no-xy" in flags,
                    dot_polar="--dot-polar" in flags,
                    lidar_features="--lidar-features" in flags,
                    seed=0,
                    mirror=False,
                    hidden=32,
                    layers=2,
                    lr=1e-3,
                    lr_patience=4,
                    resume=False,
                    epochs=1,
                    batch=1024,
                    patience=12,
                    lidar_noise=0.0,
                    velocity_noise=0.0,
                    yaw_rate_noise=0.0,
                    dropout=0.0,
                    weight_decay=0.0,
                    evals_per_epoch=1,
                )
                t.fit(self.train, self.val, out, args)
                _, saved = t.load_policy(out / "best.pt")
                self.assertEqual(len(saved["input_names"]), size)
                self.assertFalse({"v_forward", "v_left", "yaw_rate"} & set(saved["input_names"]))
                self.assertEqual("x" in saved["input_names"], name != "no_xy")
                # The training path and the driving path give the same inputs on a real row
                scale, i = t.Scale(saved=saved), np.array([700])
                past = np.maximum(i[:, None] - t.OFFSETS, self.val.start[i, None])
                row = np.concatenate([self.val.lidar[past].ravel(), self.val.state[i][0]])
                self.assertTrue(
                    np.allclose(scale.inputs(self.val, i).numpy()[0], scale.prepare(row[None])[0])
                )
                # DAgger retrains later rounds with the same flags
                self.assertEqual(
                    sorted(
                        "--" + k.replace("_", "-")
                        for k, on in saved["input_changes"].items()
                        if on
                    ),
                    sorted(flags),
                )
                # A real 2 s drive in the simulator
                result = d.drive((str(out / "best.pt"), scenario, str(Path(tmp) / "drive"), 2.0))
                self.assertEqual(result["termination"], "timeout")


if __name__ == "__main__":
    unittest.main()
