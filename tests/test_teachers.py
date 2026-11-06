"""Tests for the teacher-comparison experts (no simulator): WindowExpert's option paths and
choice, the registry."""

import unittest

import numpy as np

from comp0245.sim.obstacle_env import BestRRTExpert, VisibleRRTExpert, WindowExpert


def window():
    expert = WindowExpert.__new__(WindowExpert)
    expert.options = np.array([(v, w) for v in WindowExpert.speeds for w in WindowExpert.turns])
    return expert


class WindowTests(unittest.TestCase):
    def test_paths(self):
        """Straight option ends speed x horizon ahead; a turning option stays on its circle."""
        expert = window()
        ts, pts, th = expert.option_paths(np.zeros(2), 0.0)
        straight = int(
            np.flatnonzero((expert.options[:, 0] == 0.25) & (expert.options[:, 1] == 0.0))[0]
        )
        np.testing.assert_allclose(pts[straight, -1], [0.25 * expert.horizon, 0.0], atol=1e-9)
        k = int(np.flatnonzero((expert.options[:, 0] == 0.2) & (expert.options[:, 1] == 0.3))[0])
        radius = 0.2 / 0.3
        np.testing.assert_allclose(
            np.linalg.norm(pts[k] - [0.0, radius], axis=1), radius, atol=1e-9
        )
        stop = int(
            np.flatnonzero((expert.options[:, 0] == 0.0) & (expert.options[:, 1] == 0.0))[0]
        )
        np.testing.assert_allclose(pts[stop], 0.0, atol=1e-12)

    def test_choice(self):
        """The safe option with the best score wins; unsafe ones never win while a safe one
        exists; with none safe, the largest gap wins."""
        expert = window()
        n = len(expert.options)
        progress = np.zeros(n)
        fast = int(
            np.flatnonzero((expert.options[:, 0] == 0.25) & (expert.options[:, 1] == 0.0))[0]
        )
        progress[fast] = 1.0
        still, movers = np.full(n, 1.0), np.full(n, np.inf)
        self.assertEqual(expert.choose(still, movers, progress), fast)
        still[fast] = 0.0  # the fast option would hit something still
        self.assertNotEqual(expert.choose(still, movers, progress), fast)
        still[fast], movers[fast] = 1.0, 0.1  # or pass a mover closer than mover_gap
        self.assertNotEqual(expert.choose(still, movers, progress), fast)
        still[:] = -0.1
        still[3] = 0.0
        self.assertEqual(expert.choose(still, movers, progress), 3)

    def test_registry_and_variants(self):
        self.assertTrue(issubclass(WindowExpert, VisibleRRTExpert))
        self.assertEqual(BestRRTExpert.searches, 24)
        self.assertGreater(BestRRTExpert.shorten_iters, VisibleRRTExpert.shorten_iters)


if __name__ == "__main__":
    unittest.main()
