"""Configurable lidar ray count (A3, 3 Nov): 36 rays must stay exactly as before; other counts
(72) must plumb through."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from comp0245.evaluation.eval_obstacles import LearnedBase, load_model
from comp0245.models import BASE_INPUT_NAMES
from comp0245.sim.obstacle_env import LIDAR_ANGLES, LIDAR_RANGE, lidar_angles, lidar_points
from comp0245.training.train_base_torch import (
    FILTERS,
    INPUT_NAMES,
    SCAN_GAP_S,
    SECTORS,
    RayNet,
    build_net,
    filter_sizes,
    input_names,
    lidar_names,
    noisy_copy,
    train,
)

ROOT = Path(__file__).resolve().parents[1]


class NamesAndNetworkTests(unittest.TestCase):
    def test_36_ray_names_unchanged(self):
        self.assertEqual(input_names(1), INPUT_NAMES)
        self.assertEqual(input_names(1, 36), INPUT_NAMES)
        old = INPUT_NAMES + [
            f"lidar_t-{k * SCAN_GAP_S}s_{10 * j}deg_m" for k in range(1, 3) for j in range(36)
        ]
        self.assertEqual(
            input_names(3), old
        )  # the formula used before the ray count was a setting

    def test_72_ray_names(self):
        names = input_names(3, 72)
        self.assertEqual(len(names), len(BASE_INPUT_NAMES) + 3 * 72)
        self.assertEqual(lidar_names(72)[:3], ["lidar_0deg_m", "lidar_5deg_m", "lidar_10deg_m"])
        self.assertEqual(names[-1], "lidar_t-1.0s_355deg_m")
        self.assertEqual(len(set(names)), len(names))
        self.assertEqual(lidar_names(48)[1], "lidar_7.5deg_m")

    def test_filter_sizes_keep_the_angles(self):
        self.assertEqual(
            filter_sizes(36), (5, 3)
        )  # the 3 Nov design: 5 rays x 10 deg, 3 rays per sector
        self.assertEqual(filter_sizes(72), (9, 6))
        with self.assertRaises(ValueError):
            filter_sizes(30)

    def test_36_ray_networks_unchanged(self):
        net = build_net(42, 64)
        self.assertIsInstance(net, torch.nn.Sequential)
        self.assertEqual(
            sorted(net.state_dict()),
            ["0.bias", "0.weight", "2.bias", "2.weight", "4.bias", "4.weight"],
        )
        conv = build_net(6 + 3 * 36, 16, 3, "conv", threat=True)
        shapes = {k: tuple(v.shape) for k, v in conv.state_dict().items()}
        self.assertEqual(shapes["rays.0.weight"], (FILTERS[0], 3, 5))
        self.assertEqual(shapes["rays.2.weight"], (FILTERS[1], FILTERS[0], 5))
        self.assertEqual(shapes["trunk.0.weight"], (16, 6 + FILTERS[1] * SECTORS))
        self.assertNotIn("switch.weight", shapes)

    def test_72_ray_filter_slides_by_one_sector(self):
        """Turning every scan by one sector (6 rays of 5 deg = 30 deg) turns the sector results
        by one sector."""
        torch.manual_seed(0)
        net = RayNet(n_base=6, scans=3, hidden=16, rays=72)
        lidar = torch.rand(1, 3, 72)
        turned = torch.roll(lidar, 6, dims=2)
        with torch.no_grad():
            out, out_turned = (net.rays(v).reshape(FILTERS[1], SECTORS) for v in (lidar, turned))
        torch.testing.assert_close(torch.roll(out, 1, dims=1), out_turned)
        self.assertEqual(net(torch.rand(2, 6 + 3 * 72)).shape, (2, 4))
        self.assertEqual(
            net.trunk[0].in_features, 6 + FILTERS[1] * SECTORS
        )  # same trunk size as for 36 rays


class FinalModelsTests(unittest.TestCase):
    def test_base_final_loads_unchanged(self):
        model, saved = load_model(ROOT / "weights" / "base_final.pt")
        self.assertIsInstance(model, torch.nn.Sequential)
        self.assertNotIn("rays", saved)  # saved before the setting: read as 36
        policy = LearnedBase(model, saved)
        self.assertEqual(policy.rays, 36)
        x = torch.rand(3, len(saved["input_names"]))
        plain = torch.nn.Sequential(
            *[
                type(m)(m.in_features, m.out_features) if isinstance(m, torch.nn.Linear) else m
                for m in model
            ]
        )
        plain.load_state_dict(saved["state_dict"])
        with torch.no_grad():
            torch.testing.assert_close(model(x), plain(x))

    def test_arm_final_loads(self):
        from comp0245.evaluation.eval_arm import load_arm

        loaded = load_arm(ROOT / "weights" / "arm_final.pt")
        self.assertIsNotNone(loaded)


class PlumbingTests(unittest.TestCase):
    def test_angles(self):
        np.testing.assert_array_equal(lidar_angles(36), LIDAR_ANGLES)
        self.assertAlmostEqual(lidar_angles(72)[1], np.radians(5))

    def test_lidar_points_any_count(self):
        """One object straddling ray 0 (the ring wraps) is one cluster at 36 and at 72 rays."""
        for rays in (36, 72):
            lidar = np.full(rays, LIDAR_RANGE)
            lidar[[rays - 1, 0, 1]] = 2.0
            pts, ranges = lidar_points(np.zeros(2), 0.0, lidar)
            self.assertEqual(len(pts), 1)
            self.assertEqual(len(ranges[0]), 3)
            np.testing.assert_allclose(pts[0][1], [2.0, 0.0], atol=1e-9)

    def test_noisy_copy_touches_only_the_current_scan(self):
        rng = np.random.default_rng(0)
        x = np.full((50, 9 + 72), 2.0, np.float32)
        noisy = noisy_copy(x, rng, 0.3, 0.3, rays=72)
        np.testing.assert_array_equal(noisy[:, :9], x[:, :9])
        self.assertTrue((noisy[:, 9:] != 2.0).all(axis=0).any())
        x36 = np.full(
            (50, 45 + 72), 2.0, np.float32
        )  # 36 rays, 3 scans: earlier scans stay as they were
        np.testing.assert_array_equal(noisy_copy(x36, rng, 0.3, 0.3)[:, 45:], x36[:, 45:])

    def test_learned_base_refuses_other_ray_count(self):
        n = len(input_names(1, 72))
        saved = {
            "x_mean": np.zeros(n, np.float32),
            "x_std": np.ones(n, np.float32),
            "y_mean": np.zeros(4),
            "y_std": np.ones(4),
            "rays": 72,
        }
        policy = LearnedBase(lambda x: torch.zeros(1, 4), saved)
        state = {"x": np.zeros(28), "q": np.zeros(11), "qd": np.zeros(11)}
        policy.act({**state, "lidar": np.full(72, 4.0)})
        with self.assertRaisesRegex(ValueError, "72 lidar rays"):
            policy.act({**state, "lidar": np.full(36, 4.0)})

    def test_mixed_ray_counts_refused(self):
        with tempfile.TemporaryDirectory() as d:
            for name, rays in (("a", 36), ("b", 72)):
                (Path(d) / name).mkdir()
                (Path(d) / name / "manifest.json").write_text(
                    json.dumps({"lidar_rays": rays, "episodes": []})
                )
            with self.assertRaisesRegex(ValueError, "different lidar ray counts"):
                train([Path(d) / "a"], out=Path(d) / "out", validation=Path(d) / "b")

    def test_72_ray_model_saves_and_loads(self):
        names = input_names(3, 72)
        net = build_net(len(names), 16, 3, "conv", rays=72)
        n = len(names)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.pt"
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "hidden": 16,
                    "scans": 3,
                    "encoder": "conv",
                    "rays": 72,
                    "input_names": names,
                    "columns": list(range(n)),
                    "x_mean": np.zeros(n),
                    "x_std": np.ones(n),
                    "y_mean": np.zeros(4),
                    "y_std": np.ones(4),
                },
                path,
            )
            loaded, saved = load_model(path)
        x = torch.rand(4, n)
        with torch.no_grad():
            torch.testing.assert_close(loaded(x), net(x))


if __name__ == "__main__":
    unittest.main()
