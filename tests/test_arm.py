"""Tests for the learned arm: its inputs (and later the gravity term and the IK direction)."""

import unittest

import numpy as np

from comp0245.models import ARM_INPUT_NAMES, arm_inputs


class ArmTests(unittest.TestCase):
    def test_arm_inputs(self):
        obs = np.zeros(28)
        obs[4] = 1.0  # cos(yaw) = 1: robot at the origin facing +x
        obs[5:8] = [1.2, 0.1, 0.6]  # target
        ee = np.array([1.0, 0.0, 0.5])  # hand
        x = arm_inputs(obs, ee)
        self.assertEqual(len(x), len(ARM_INPUT_NAMES))
        np.testing.assert_allclose(x[-6:-3], [0.2, 0.1, 0.1])
        np.testing.assert_allclose(x[-3:], [0.996, 0.1, 0.104])

    def test_arm_inputs_turned(self):
        obs = np.zeros(28)
        obs[3] = 1.0  # sin(yaw) = 1, cos(yaw) = 0: robot at the origin facing +y
        obs[5:8] = [0.0, 1.0, 0.5]  # target 1 m in +y
        ee = np.array([0.0, 0.8, 0.5])  # hand 0.8 m in +y
        x = arm_inputs(obs, ee)
        np.testing.assert_allclose(x[-6:-3], [0.2, 0, 0], atol=1e-9)
        np.testing.assert_allclose(x[-3:], [0.796, 0, 0.004], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
