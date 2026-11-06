"""Tests for the base scorer and model loading (fixes from Codex's 28 Oct review)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from comp0245.evaluation.eval_obstacles import SETTLE_WINDOW_S, load_model, score
from comp0245.sim.obstacle_env import standoff
from comp0245.training.train_base_torch import INPUT_NAMES

TARGET = [3.0, 0.0, 0.5]


def parked_episode(folder, yaw_rate=0.0, early_offset=0.0):
    """40 s at 100 rows/s, parked exactly at the reach-zone distance facing the target.
    early_offset (m) moves the base back for the rows from 1.5 s to 1 s before the end (inside
    the settle window)."""
    t = np.arange(0, 40, 0.01)
    obs = np.zeros((len(t), 28))
    obs[:, 0] = TARGET[0] - standoff(TARGET[2])
    obs[:, 4] = 1.0  # cos(yaw) = 1: facing +x, at the target
    obs[:, 27] = yaw_rate
    obs[(t > 38.5) & (t < 39.0), 0] -= early_offset
    np.savez(folder / "trajectory.npz", observations=obs, time_s=t)
    return {
        "final_standoff_error_m": 0.0,
        "final_facing_error_rad": 0.0,
        "contact_steps": 0,
        "min_clearance_m": 1.0,
        "base_path_length_m": 3.0,
    }


class ScoringTests(unittest.TestCase):
    def test_parked_and_still_passes(self):
        with tempfile.TemporaryDirectory() as d:
            r = score(Path(d), {"id": "x", "target": TARGET}, parked_episode(Path(d)))
        self.assertTrue(r["strict"] and r["loose"])

    def test_spinning_in_place_fails(self):
        with tempfile.TemporaryDirectory() as d:
            r = score(
                Path(d), {"id": "x", "target": TARGET}, parked_episode(Path(d), yaw_rate=1.0)
            )
        self.assertFalse(r["strict"] or r["loose"])

    def test_parking_must_hold_for_the_whole_window(self):
        self.assertGreaterEqual(SETTLE_WINDOW_S, 1.5)
        with tempfile.TemporaryDirectory() as d:
            r = score(
                Path(d), {"id": "x", "target": TARGET}, parked_episode(Path(d), early_offset=0.1)
            )
        self.assertFalse(r["strict"])
        self.assertTrue(r["loose"])
        self.assertAlmostEqual(r["worst_distance_error_m"], 0.1)

    def test_contact_fails(self):
        with tempfile.TemporaryDirectory() as d:
            summary = {**parked_episode(Path(d)), "contact_steps": 1}
            r = score(Path(d), {"id": "x", "target": TARGET}, summary)
        self.assertFalse(r["strict"] or r["loose"])


class LoadModelTests(unittest.TestCase):
    def save(self, folder, names, columns):
        n = len(names)
        model = nn.Sequential(
            nn.Linear(n, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hidden": 8,
                "input_names": names,
                "columns": columns,
                "x_mean": np.zeros(n),
                "x_std": np.ones(n),
                "y_mean": np.zeros(4),
                "y_std": np.ones(4),
            },
            folder / "model.pt",
        )
        return folder / "model.pt"

    def test_matching_inputs_load_on_cpu(self):
        columns = [i for i, n in enumerate(INPUT_NAMES) if n != "yaw_rate_rad_s"]
        with tempfile.TemporaryDirectory() as d:
            model, _ = load_model(self.save(Path(d), [INPUT_NAMES[i] for i in columns], columns))
        self.assertEqual(next(model.parameters()).device.type, "cpu")

    def test_mismatched_inputs_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.save(Path(d), list(reversed(INPUT_NAMES)), list(range(len(INPUT_NAMES))))
            with self.assertRaises(ValueError):
                load_model(path)


if __name__ == "__main__":
    unittest.main()
