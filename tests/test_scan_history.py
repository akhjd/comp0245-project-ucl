"""Stacked lidar scans: what the base model sees while driving must match what it was trained
on."""

import unittest

import numpy as np
import torch

from comp0245.evaluation.eval_obstacles import LearnedBase
from comp0245.models import base_inputs
from comp0245.training.train_base_torch import SCAN_GAP_S, STRIDE, input_names, stack_scans


class Recorder:
    """Stands in for the network and keeps the inputs it was given."""

    def __init__(self):
        self.seen = []

    def __call__(self, x):
        self.seen.append(x[0].numpy().copy())
        return torch.zeros(1, 4)


class ScanHistoryTests(unittest.TestCase):
    def test_stack_scans(self):
        lidar = np.arange(12.0).reshape(4, 3)
        np.testing.assert_array_equal(stack_scans(lidar, 1, 2), lidar)
        np.testing.assert_array_equal(stack_scans(lidar, 2, 2)[:, 3:], lidar[[0, 0, 0, 1]])
        self.assertEqual(stack_scans(lidar, 3, 2).shape, (4, 9))

    def test_names_count(self):
        self.assertEqual(len(input_names(1)), 45)
        self.assertEqual(len(input_names(3)), 117)

    def test_driving_matches_training(self):
        rng = np.random.default_rng(0)
        steps = 2500  # 2.5 s at 1 kHz, so the 1 s window fills up
        lidar = rng.uniform(0.5, 4.0, (steps, 36))
        obs = rng.uniform(-1.0, 1.0, (steps, 28))
        n = len(input_names(3))
        saved = {
            "x_mean": np.zeros(n, np.float32),
            "x_std": np.ones(n, np.float32),
            "y_mean": np.zeros(4),
            "y_std": np.ones(4),
            "scans": 3,
        }
        recorder = Recorder()
        policy = LearnedBase(recorder, saved)
        for k in range(steps):
            policy.act({"x": obs[k], "lidar": lidar[k], "q": np.zeros(11), "qd": np.zeros(11)})
        rows = obs[::STRIDE], lidar[::STRIDE]
        trained = np.hstack(
            [
                np.array([base_inputs(r) for r in rows[0]]),
                stack_scans(rows[1], 3, int(round(SCAN_GAP_S * 1000 / STRIDE))),
            ]
        ).astype(np.float32)
        np.testing.assert_array_equal(np.array(recorder.seen)[::STRIDE], trained)


if __name__ == "__main__":
    unittest.main()
