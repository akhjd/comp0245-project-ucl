"""Tests for the menu output and evade weighting (no simulator)."""

import unittest

import numpy as np
import torch

from comp0245.sim.obstacle_env import WindowExpert
from comp0245.training.train_base_torch import (
    HOLD,
    MENU_SPEEDS,
    MENU_TURNS,
    TRACK,
    WHEEL_RADIUS,
    build_net,
    menu_labels,
    menu_loss,
    menu_options,
    speed_turn,
)


def wheels(v, w):
    left, right = (v - w * TRACK / 2) / WHEEL_RADIUS, (v + w * TRACK / 2) / WHEEL_RADIUS
    return np.array([[right, left, left, right]])


class MenuTests(unittest.TestCase):
    def test_same_menu_as_the_teacher(self):
        self.assertEqual(tuple(MENU_SPEEDS), tuple(WindowExpert.speeds))
        self.assertEqual(tuple(MENU_TURNS), tuple(WindowExpert.turns))
        self.assertEqual(len(menu_options()), 64)
        self.assertEqual(HOLD, 63)

    def test_labels(self):
        """An on-menu command gets its own option and no correction; an off-menu one the nearest
        option and the gap."""
        choice, gap = menu_labels(wheels(0.2, -0.45))
        self.assertTrue(np.allclose(menu_options()[choice[0]], [0.2, -0.45]))
        np.testing.assert_allclose(gap, 0.0, atol=1e-5)
        choice, gap = menu_labels(
            wheels(0.02, 0.05)
        )  # a slow parking command: nearest is "stop", not hold
        self.assertTrue(np.allclose(menu_options()[choice[0]], [0.0, 0.0]))
        self.assertNotEqual(choice[0], HOLD)
        np.testing.assert_allclose(gap[0], [0.02 / 0.05, 0.05 / 0.15], atol=1e-5)
        v, w = speed_turn(wheels(0.1, 0.3))
        self.assertAlmostEqual(float(v[0]), 0.1, places=6)
        self.assertAlmostEqual(float(w[0]), 0.3, places=6)

    def test_hold_is_exactly_still(self):
        """Exactly-zero commands are labelled hold; a net that picks hold drives exactly 0
        whatever its correction says."""
        choice, gap = menu_labels(np.zeros((3, 4)))
        self.assertTrue((choice == HOLD).all())
        np.testing.assert_array_equal(gap, 0.0)
        net = build_net(9 + 36, 16, 1, "conv", menu=True)
        with torch.no_grad():
            net.wheels.bias.zero_()
            net.wheels.bias[HOLD] = 100.0  # always picks hold
            net.wheels.bias[64:] = 3.0  # with large corrections everywhere
            out = net(torch.randn(5, 45)).numpy()
        np.testing.assert_array_equal(out, 0.0)

    def test_commits_to_one_option(self):
        """Trained on the same input labelled hard left half the time and hard right the other
        half, the menu output drives one of them at full turn, never the straight-line average."""
        torch.manual_seed(0)
        net = build_net(9 + 36, 16, 1, "conv", menu=True)
        x = torch.zeros(64, 45)
        y = np.concatenate([wheels(0.2, 0.6)] * 32 + [wheels(0.2, -0.6)] * 32)
        choice, gap = menu_labels(y)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        for _ in range(200):
            loss = menu_loss(
                *net.choices(net.features(x)), torch.as_tensor(choice), torch.as_tensor(gap)
            ).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            out = net(x[:1]).numpy()  # y_mean 0, y_std 1 here: raw wheel speeds
        v, w = speed_turn(out)
        self.assertAlmostEqual(abs(float(w[0])), 0.6, places=2)
        self.assertAlmostEqual(float(v[0]), 0.2, places=2)


if __name__ == "__main__":
    unittest.main()
