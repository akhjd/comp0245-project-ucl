"""Tests for the "parked" base score (eval_obstacles.score): inside the measured reach area,
looking towards the target, still for the last 2 s, no contact."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from comp0245.evaluation.eval_obstacles import FACING_LIMIT, REACH_MIN_M, reach_limit, score


def run(distance, facing_deg=0.0, contact=0, speed=0.0, target_z=0.58):
    """Score a still base 4 s long at `distance` m from the target, turned `facing_deg` away
    from it."""
    rows = 400
    obs = np.zeros((rows, 28))
    yaw = np.radians(facing_deg)  # target straight along +x, base at the origin
    obs[:, 3], obs[:, 4] = np.sin(yaw), np.cos(yaw)
    obs[:, 22] = speed
    target = [distance, 0.0, target_z]
    summary = {
        "final_standoff_error_m": 0.0,
        "final_facing_error_rad": abs(yaw),
        "contact_steps": contact,
        "min_clearance_m": 0.5,
        "base_path_length_m": 3.0,
    }
    with tempfile.TemporaryDirectory() as d:
        np.savez(Path(d) / "trajectory.npz", observations=obs, time_s=np.arange(rows) * 0.01)
        return score(Path(d), {"id": "x", "target": target}, summary)


class ParkedTests(unittest.TestCase):
    def test_inside_the_area_facing_it(self):
        for d in (REACH_MIN_M, 0.6, 0.93, reach_limit(0.58) - 0.01):
            self.assertTrue(run(d)["parked"], d)
        self.assertTrue(run(0.93, facing_deg=25)["parked"])  # does not have to face it exactly

    def test_not_looking_at_it(self):
        self.assertGreater(np.radians(40), FACING_LIMIT)
        self.assertFalse(run(0.93, facing_deg=40)["parked"])

    def test_contact_fails(self):
        self.assertFalse(run(0.93, contact=1)["parked"])

    def test_too_far_or_too_close(self):
        self.assertFalse(run(reach_limit(0.58) + 0.02)["parked"])
        self.assertFalse(run(REACH_MIN_M - 0.05)["parked"])

    def test_reach_depends_on_facing(self):
        # measured edges (runs/reach-area-fine): straight ahead 1.04 / 1.06 / 1.00 m, 30 deg off
        # 1.02 / 1.02 / 0.98 m
        for z, ahead, side in ((0.36, 1.04, 1.02), (0.58, 1.06, 1.02), (0.80, 1.00, 0.98)):
            self.assertLessEqual(reach_limit(z, 0.0), ahead)
            self.assertLessEqual(
                reach_limit(z), side + 0.002
            )  # the 83% rule (3 Nov) is within 2 mm of these
        d = reach_limit(0.58) + 0.01  # past the worst-case edge, inside the straight-ahead one
        r = run(d)
        self.assertTrue(r["parked"])
        self.assertFalse(r["parked_worst_case_reach"])
        self.assertFalse(run(d, facing_deg=29)["parked"])

    def test_must_be_still(self):
        self.assertFalse(run(0.93, speed=0.05)["parked"])

    def test_far_from_the_experts_ring_still_counts(self):
        """The old strict rule wanted the expert's distance (about 0.93 m) within 5 cm; parked
        does not."""
        r = run(0.7)
        self.assertTrue(r["parked"])
        self.assertFalse(r["strict"])


if __name__ == "__main__":
    unittest.main()
