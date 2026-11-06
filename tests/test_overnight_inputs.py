"""Tests for the 5 Nov overnight additions: the "target blocked?" inputs
(train_base_torch.target_ray), the same inputs in training (load) and driving (LearnedBase),
dropout / weight decay models, and the lidar range setting."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from comp0245.paths import SETTINGS_FILE
from comp0245.training import train_base_torch as tb
from tests.test_groups import fake_dataset


class TargetRayTests(unittest.TestCase):
    def test_reads_the_ray_towards_the_target(self):
        rays = 144
        lidar = np.full((1, rays), 10.0)
        lidar[0, 36] = 2.0  # ray 36 of 144 points 90 deg to the left
        base = np.zeros((1, 9))
        base[0, :2] = [0.0, 3.0]  # target 3 m straight to the left
        seen, gap = tb.target_ray(base, lidar)[0]
        self.assertAlmostEqual(seen, 2.0, places=5)
        self.assertAlmostEqual(gap, -1.0, places=5)  # a wall 1 m before the target

    def test_interpolates_between_rays(self):
        lidar = np.array([[4.0, 8.0, 4.0, 4.0]])  # 4 rays, 90 deg apart
        base = np.zeros((1, 9))
        base[0, :2] = [1.0, 1.0]  # 45 deg: halfway between ray 0 and ray 1
        self.assertAlmostEqual(tb.target_ray(base, lidar)[0, 0], 6.0, places=5)

    def test_training_and_driving_inputs_match(self):
        """The row train_base_torch.load builds equals what LearnedBase computes from the same
        state."""
        from comp0245.evaluation.eval_obstacles import LearnedBase

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "data"
            fake_dataset(d, [("ova_000", "train"), ("ova_001", "validation")], rays=144)
            tb.train(
                [str(d)],
                [],
                Path(tmp) / "m",
                True,
                epochs=1,
                validation=str(d),
                scans=3,
                hidden=16,
                encoder="conv",
                target_ray=True,
                dropout=0.2,
                weight_decay=0.01,
            )
            from comp0245.evaluation.eval_obstacles import load_model

            model, saved = load_model(Path(tmp) / "m" / "model.pt")
            self.assertTrue(saved["target_ray"])
            self.assertEqual(
                saved["input_names"][6:8], tb.TARGET_RAY_NAMES
            )  # after the 6 kept base features
            x, *_ = tb.load("train", d, scans=3, target_ray_=True)
            z = np.load(d / "ova_000" / "trajectory.npz")
            policy = LearnedBase(model, saved)
            captured = {}
            real = policy.model.forward
            policy.model.forward = lambda xt: captured.setdefault("x", xt) is not None and real(xt)
            for k in range(0, 1):  # first row: all three scans are the current one in both paths
                state = {
                    "x": z["observations"][k],
                    "lidar": z["lidar"][k],
                    "q": np.zeros(11),
                    "qd": np.zeros(11),
                }
                policy.act(state)
            row = captured["x"][0].numpy() * saved["x_std"] + saved["x_mean"]
            np.testing.assert_allclose(row, x[0][saved["columns"]], rtol=1e-4, atol=1e-4)

    def test_dropout_model_is_deterministic_when_driving(self):
        net = tb.build_net(
            9 + 2 + 144 * 3, 32, scans=3, encoder="conv", threat=True, rays=144, dropout=0.3
        )
        net.eval()
        xt = torch.randn(4, 9 + 2 + 144 * 3)
        torch.testing.assert_close(net(xt), net(xt))
        self.assertTrue(any(isinstance(m, torch.nn.Dropout) for m in net.trunk))
        old = tb.build_net(
            9 + 144 * 3, 32, scans=3, encoder="conv", threat=True, rays=144
        )  # dropout 0: old layers
        self.assertFalse(any(isinstance(m, torch.nn.Dropout) for m in old.trunk))
        self.assertEqual(len(old.trunk), 4)


class LidarRangeTests(unittest.TestCase):
    def test_range_setting_reaches_further(self):
        import contextlib
        import io

        from comp0245.pipeline import read_json
        from comp0245.sim.obstacle_env import ObstacleRobot

        config = read_json(SETTINGS_FILE)
        s = {
            "id": "r",
            "split": "validation",
            "seed": 0,
            "start": [0.0, 0.0, 0.0],
            "target": [3.0, 0.0, 0.6],
            "park": [2.0, 0.0],
            "obstacles": [],
            "room": [-7.0, -7.0, 7.0, 7.0],
        }
        for reach, expect in ((4.0, 4.0), (10.0, 7.0)):
            cfg = {**config, "lidar_rays": 144, "lidar_range": reach}
            with contextlib.redirect_stdout(io.StringIO()):
                r = ObstacleRobot(cfg, s)
            try:
                lid = r.lidar(r.measure())
                self.assertEqual(len(lid), 144)
                self.assertAlmostEqual(
                    float(lid[0]), expect, delta=0.05
                )  # wall 7 m ahead: seen only with 10 m
            finally:
                r.close()


if __name__ == "__main__":
    unittest.main()
