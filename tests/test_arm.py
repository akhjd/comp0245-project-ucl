"""Tests for the learned arm: its inputs (and later the gravity term and the IK direction)."""
import unittest
from pathlib import Path
import numpy as np
import pinocchio as pin
from arm_data import _data, _model, DAMPING, EE_FRAME, ik_direction, arm_inputs_b, ARM_B_INPUT_NAMES, arm_input_fn
from models import arm_inputs, ARM_INPUT_NAMES


class ArmTests(unittest.TestCase):
    def test_arm_inputs(self):
        obs = np.zeros(28)
        obs[4] = 1.0                    # cos(yaw) = 1: robot at the origin facing +x
        obs[5:8] = [1.2, 0.1, 0.6]      # target
        ee = np.array([1.0, 0.0, 0.5])  # hand
        x = arm_inputs(obs, ee)
        self.assertEqual(len(x), len(ARM_INPUT_NAMES))
        np.testing.assert_allclose(x[-6:-3], [0.2, 0.1, 0.1])
        np.testing.assert_allclose(x[-3:], [0.996, 0.1, 0.104])

    def test_arm_inputs_turned(self):
        obs = np.zeros(28)
        obs[3] = 1.0                    # sin(yaw) = 1, cos(yaw) = 0: robot at the origin facing +y
        obs[5:8] = [0.0, 1.0, 0.5]      # target 1 m in +y
        ee = np.array([0.0, 0.8, 0.5])  # hand 0.8 m in +y
        x = arm_inputs(obs, ee)
        np.testing.assert_allclose(x[-6:-3], [0.2, 0, 0], atol=1e-9)
        np.testing.assert_allclose(x[-3:], [0.796, 0, 0.004], atol=1e-9)

    def test_ik_direction(self):
        q_arm = np.array([0, 1.57, 0, 1, 0, 1, 0])      # a bent pose, away from singular ones
        e = np.array([0.01, 0, 0])                      # move the hand 1 cm forward
        q = pin.neutral(_model)
        q[:7] = q_arm
        J = pin.computeFrameJacobian(_model, _data, q, EE_FRAME, pin.LOCAL_WORLD_ALIGNED)[:3, :7]
        speeds = ik_direction(q_arm, e)
        self.assertEqual(speeds.shape, (7,))
        np.testing.assert_allclose(J @ speeds, e, atol=1e-3)   # those joint speeds really move the hand along e

    def test_model_b_inputs(self):
        obs = np.zeros(28)
        obs[4] = 1.0
        obs[5:8] = [1.2, 0.1, 0.6]
        obs[8:15] = [0, 1.57, 0, 1, 0, 1, 0]
        ee = np.array([1.0, 0.0, 0.5])
        x = arm_inputs_b(obs, ee)
        self.assertEqual(len(x), len(ARM_B_INPUT_NAMES))
        np.testing.assert_allclose(x[:24], arm_inputs(obs, ee))      # model A's inputs come first, unchanged
        np.testing.assert_allclose(x[24:], ik_direction(obs[8:15], [0.2, 0.1, 0.1]))
        self.assertIs(arm_input_fn(ARM_INPUT_NAMES), arm_inputs)
        self.assertIs(arm_input_fn(ARM_B_INPUT_NAMES), arm_inputs_b)

    def test_final_arm_still_loads(self):
        from eval_arm import load_arm
        _, saved = load_arm(Path(__file__).resolve().parents[1]/"weights/arm_final.pt")
        self.assertEqual(list(saved["input_names"]), ARM_INPUT_NAMES)

if __name__ == "__main__":
    unittest.main()