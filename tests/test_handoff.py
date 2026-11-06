"""Tests for the arm hand-off in eval_arm.LearnedArm (5 Nov): the arm waits for the full parking rule, and a folded
arm unfolds to the normal start pose before the learned arm takes over."""
import unittest
import numpy as np
from eval_arm import LearnedArm, ARM_START, UNFOLD_RATE
from eval_obstacles import ARM_FOLD, SETTLE_WINDOW_S, reach_limit

TARGET = [2.0, 0.0, 0.58]


class StillBase:
    """A learned base stand-in (no "arrived"): always commands zero wheels."""
    def act(self, state):
        return np.zeros(11)


def state(base_xy, yaw=0.0, speed=0.0, q=ARM_START, qd=0.0):
    x = np.zeros(28)
    x[0:2], x[3], x[4], x[5:8], x[22] = base_xy, np.sin(yaw), np.cos(yaw), TARGET, speed
    q_all, qd_all = np.zeros(11), np.zeros(11)
    q_all[4:11], qd_all[4:11] = q, qd
    return {"x": x, "base": np.array([*base_xy, 0.0]), "yaw": yaw, "q": q_all, "qd": qd_all}


def arm(unfold=False):
    return LearnedArm(None, {}, TARGET, StillBase(), unfold=unfold)


def steps_until_parked(policy, s, limit=6000):
    for k in range(limit):
        if policy.parked(s):
            return k
        policy.t += 0.001
    return None


class HandoffTests(unittest.TestCase):
    def test_waits_for_the_whole_window(self):
        inside = [TARGET[0] - 0.9*reach_limit(TARGET[2], 0.0), 0.0]
        k = steps_until_parked(arm(), state(inside))
        # 2 s of time before counting starts, then the full settle window
        self.assertAlmostEqual(k*0.001, 2.0 + SETTLE_WINDOW_S, delta=0.01)

    def test_refuses_outside_the_reach_area(self):
        too_far = [TARGET[0] - reach_limit(TARGET[2], 0.0) - 0.05, 0.0]
        self.assertIsNone(steps_until_parked(arm(), state(too_far)))

    def test_refuses_when_not_facing_or_moving(self):
        inside = [TARGET[0] - 0.8, 0.0]
        self.assertIsNone(steps_until_parked(arm(), state(inside, yaw=np.radians(35))))
        self.assertIsNone(steps_until_parked(arm(), state(inside, speed=0.05)))

    def test_unfold_ramps_to_the_start_pose(self):
        policy = arm(unfold=True)
        policy.q_hold = np.array(ARM_FOLD, float)
        k = 0
        while policy.unfolding(np.zeros(7)):
            k += 1
        np.testing.assert_allclose(policy.q_hold, ARM_START, atol=1e-9)
        slowest = np.abs(np.array(ARM_START) - ARM_FOLD).max()/UNFOLD_RATE
        self.assertAlmostEqual(k*0.001, slowest, delta=0.01)
        # the learned arm waits while the joints are still moving
        self.assertTrue(policy.unfolding(np.full(7, 0.1)))


if __name__ == "__main__":
    unittest.main()
