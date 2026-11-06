"""Tests for the learned ray filter (RayNet, encoder "conv") and the training-only threat labels."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from eval_obstacles import load_model
from train_base_torch import RayNet, build_net, input_names, threat_labels, VELOCITY, SECTORS, FILTERS, STRIDE


class RayNetTests(unittest.TestCase):
    def test_plain_network_unchanged(self):
        net = build_net(42, 64)
        self.assertIsInstance(net, torch.nn.Sequential)
        self.assertEqual(sorted(net.state_dict()), ["0.bias", "0.weight", "2.bias", "2.weight", "4.bias", "4.weight"])

    def test_filter_slides_round_the_robot(self):
        """Turning every scan by one sector (3 rays) turns the filter's sector results by one sector: the same
        weights look at every angle, and position in the output says the direction."""
        torch.manual_seed(0)
        net = RayNet(n_base=6, scans=3, hidden=16)
        lidar = torch.rand(1, 3, 36)
        turned = torch.roll(lidar, 3, dims=2)
        with torch.no_grad():
            out, out_turned = (net.rays(v).reshape(FILTERS[1], SECTORS) for v in (lidar, turned))
        torch.testing.assert_close(torch.roll(out, 1, dims=1), out_turned)

    def test_saved_model_loads(self):
        names = input_names(3)
        columns = [i for i, n in enumerate(names) if n not in VELOCITY]
        net = build_net(len(columns), 16, 3, "conv", threat=True)
        n = len(columns)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"model.pt"
            torch.save({"state_dict": net.state_dict(), "hidden": 16, "scans": 3, "encoder": "conv", "threat": True,
                        "input_names": [names[i] for i in columns], "columns": columns,
                        "x_mean": np.zeros(n), "x_std": np.ones(n), "y_mean": np.zeros(4), "y_std": np.ones(4)}, path)
            loaded, _ = load_model(path)
        x = torch.rand(5, n)
        with torch.no_grad():
            torch.testing.assert_close(loaded(x), net(x))
        self.assertEqual(loaded(x).shape, (5, 4))


class ThreatLabelTests(unittest.TestCase):
    def test_approaching_mover(self):
        """Robot still at the origin facing +y; a mover 2 m ahead comes at 0.5 m/s; a still obstacle sits 0.3 m away
        and must be ignored. Rows are STRIDE ms apart (already strided: stride 1)."""
        dt, rows = STRIDE/1000, 600
        t = np.arange(rows)*dt
        obs = np.zeros((rows, 28))
        obs[:, 3] = 1.0                                    # sin(yaw) = 1: facing +y
        mover = np.stack([np.zeros(rows), 2.0 - 0.5*t], axis=1)
        still = np.tile([0.3, 0.0], (rows, 1))
        z = {"observations": obs, "base_m": np.zeros((rows, 3)), "obstacle_xy": np.stack([mover, still], axis=1)}
        z = type("Npz", (dict,), {"files": property(lambda self: list(self))})(z)
        labels = threat_labels(z, [True, False], 1)
        self.assertEqual(labels[0, 0], 1.0)                # within 3 s it gets to 0.5 m
        np.testing.assert_allclose(labels[0, 1:], [2.0, 0.0, -0.5, 0.0], atol=1e-6)   # ahead, coming closer
        far = labels[:, 0] == 1.0
        self.assertTrue(far.all())                         # from row 0 on it is always within 1 m within 3 s
        z["obstacle_xy"] = np.stack([mover + [10.0, 0.0], still], axis=1)
        self.assertFalse(threat_labels(z, [True, False], 1)[:, 0].any())   # 10 m to the side: never a threat
        self.assertFalse(threat_labels(z, [False, False], 1).any())        # no movers: all zeros


if __name__ == "__main__":
    unittest.main()
