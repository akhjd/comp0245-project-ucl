"""Tests for the door memory in preprocess.py (feature set v2), on made-up scans from the ray caster in test_preprocess.

    python -m unittest tests.test_door_memory
"""
import unittest
import numpy as np
import preprocess as pp
from tests.test_preprocess import DOOR_WALLS, NO_DOTS, ROOM, cast

V2 = pp.FEATURE_SETS["v2"]


def value(features, name):
    return float(features[V2.index(name)])


def world(f, prefix, pose):
    """A reported doorway (distance, sin, cos from the robot) in world coordinates."""
    x, y, yaw = pose
    d, si, co = value(f, f"{prefix}_distance"), value(f, f"{prefix}_sin"), value(f, f"{prefix}_cos")
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([x, y]) + d*np.array([c*co - s*si, s*co + c*si])


def drive(pre, poses, boxes, dots=NO_DOTS, dt=0.01):
    """Feed one scan per pose; returns the last features."""
    for i, pose in enumerate(poses):
        f = pre.step(i*dt, cast(pose[:2], pose[2], boxes), pose, dots)
    return f


class DoorMemoryTests(unittest.TestCase):
    def test_v1_is_unchanged(self):
        self.assertEqual(len(pp.MLP_NAMES), 61)
        self.assertEqual(pp.FEATURE_SETS["v1"], pp.MLP_NAMES)
        self.assertEqual(V2[:61], pp.MLP_NAMES)
        self.assertFalse(set(pp.MEMORY) & set(pp.MLP_NAMES))
        scan, pose = cast((0, 0), 0.2, ROOM + DOOR_WALLS), (0, 0, 0.2)
        self.assertTrue(np.array_equal(pp.Preprocessor(pp.MLP_NAMES).step(0, scan, pose, NO_DOTS),
                                       pp.Preprocessor(V2).step(0, scan, pose, NO_DOTS)[:61]))

    def test_remembers_the_doorway_out_of_range(self):
        pre = pp.Preprocessor(V2)
        near = [(0.0, 0.1*np.sin(i/5), 0.05*i) for i in range(20)]               # looks at the doorway at (2, 0) for 0.2 s
        away = [(-0.15*i, 0.0, 1.0) for i in range(1, 21)]                       # then backs off to x = -3, 5 m from it
        f = drive(pre, near, ROOM + DOOR_WALLS)
        self.assertEqual(value(f, "door_seen"), 1)
        f = drive(pre, away, ROOM + DOOR_WALLS)
        self.assertEqual(value(f, "opening_seen"), 0)                            # this scan cannot see it any more...
        self.assertEqual(value(f, "door_seen"), 1)                               # ...but the memory still has it
        self.assertEqual(value(f, "doors_remembered"), 1)
        self.assertLess(np.linalg.norm(world(f, "door", away[-1]) - (2.0, 0.0)), 0.2)
        self.assertGreater(value(f, "door_age_s"), 0.05)
        angle = np.arctan2(value(f, "door_dir_sin2"), value(f, "door_dir_cos2"))/2 + away[-1][2]
        self.assertLess(abs(np.cos(angle)), 0.2)                                 # the wall runs along y

    def test_dot_behind_the_wall_is_across(self):
        pre = pp.Preprocessor(V2)
        pose = (0.0, 0.0, 0.0)
        dots = NO_DOTS.copy()
        dots[0] = [3.5, 1.5, 0.5, 1]                                            # beyond the wall at x = 2
        dots[1] = [-1.0, 1.0, 0.5, 1]                                           # this side of it
        f = drive(pre, [pose]*5, ROOM + DOOR_WALLS, dots)
        self.assertEqual(value(f, "dot0_door_seen"), 1)
        self.assertEqual(value(f, "dot0_door_across"), 1)
        self.assertLess(np.linalg.norm(world(f, "dot0_door", pose) - (2.0, 0.0)), 0.2)
        self.assertEqual(value(f, "dot1_door_across"), 0)
        self.assertEqual(value(f, "dot2_door_seen"), 0)                          # unused slot

    def test_lined_up_furniture_is_not_a_doorway(self):
        furniture = [(1.95, -1.5, 2.75, -0.7), (1.95, 0.7, 2.75, 1.5)]          # 0.8 m faces in line, 1.4 m apart
        pre, door_like = pp.Preprocessor(V2), []
        for i in range(30):
            pose = (0.0, 0.0, 0.03*i)
            f = pre.step(i/100, cast(pose[:2], pose[2], ROOM + furniture), pose, NO_DOTS)
            door_like.append(value(f, "opening_door_like"))
        self.assertGreater(sum(door_like), 0)                                    # it does look like a doorway in one scan...
        self.assertEqual(value(f, "door_seen"), 0)                               # ...but no wall piece is long enough

    def test_reset_forgets(self):
        pre = pp.Preprocessor(V2)
        drive(pre, [(0.0, 0.0, 0.0)]*5, ROOM + DOOR_WALLS)
        self.assertTrue(pre.doors)
        pre.reset()
        self.assertFalse(pre.doors)
        f = pre.step(0.0, cast((-3, 0), 0, ROOM + DOOR_WALLS), (-3, 0, 0), NO_DOTS)
        self.assertEqual(value(f, "door_seen"), 0)


if __name__ == "__main__":
    unittest.main()
