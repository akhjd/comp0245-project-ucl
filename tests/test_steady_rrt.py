"""Tests for SteadyRRTExpert's side rule (no simulator needed)."""
import unittest
import numpy as np
from obstacle_env import SteadyRRTExpert


class SideTests(unittest.TestCase):
    def test_side_of_the_line(self):
        p, goal = np.zeros(2), np.array([4.0, 0.0])
        left = np.array([[0, 1, 1, 0, 0, 0], [0, 3, 1, 0, 0, 0]], float)     # rows: t, x, y, yaw, v, w
        self.assertAlmostEqual(SteadyRRTExpert.side(left, p, goal), 1.0)
        self.assertAlmostEqual(SteadyRRTExpert.side(left*[1, 1, -1, 1, 1, 1], p, goal), -1.0)

    def test_wanted_side(self):
        expert = SteadyRRTExpert.__new__(SteadyRRTExpert)
        p, goal = np.zeros(2), np.array([4.0, 0.0])
        self.assertEqual(expert.wanted_side(p, 0.3, goal), 1.0)      # facing a little left of the goal: left
        self.assertEqual(expert.wanted_side(p, -0.3, goal), -1.0)
        expert.last_side = -1.0                                       # once it has chosen, it keeps that side
        self.assertEqual(expert.wanted_side(p, 0.3, goal), -1.0)


if __name__ == "__main__":
    unittest.main()
