"""Tests for comp0245/preprocess.py on made-up scans from a small ray caster (boxes and circles,
like the simulator's lidar).

    python -m unittest tests.test_preprocess
"""

import unittest

import numpy as np

from comp0245 import preprocess as pp


def cast(robot, yaw, boxes=(), circles=()):
    """36 lidar distances from robot (x, y) with heading yaw: boxes [(xmin, ymin, xmax, ymax)],
    circles [(x, y, r)]. Like obstacle_env: rays from 0.5 m to 4 m, 4 m if nothing is hit."""
    out = np.full(pp.RAYS, pp.RANGE)
    o = np.asarray(robot, dtype=float)
    for k, a in enumerate(pp.ANGLES + yaw):
        d = np.array([np.cos(a), np.sin(a)])
        best = pp.RANGE
        for x0, y0, x1, y1 in boxes:
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (np.array([x0, y0]) - o) / d, (np.array([x1, y1]) - o) / d
            lo, hi = np.nanmax(np.minimum(t1, t2)), np.nanmin(np.maximum(t1, t2))
            if hi >= max(lo, 0.5) and lo <= best:
                best = min(best, max(lo, 0.5))
        for cx, cy, r in circles:
            f = o - (cx, cy)
            b, c = f @ d, f @ f - r * r
            if b * b - c >= 0:
                t = -b - np.sqrt(b * b - c)
                if 0.5 <= t < best:
                    best = t
        out[k] = best
    return out


ROOM = [(-4, -4.1, 4, -4), (-4, 4, 4, 4.1), (-4.1, -4, -4, 4), (4, -4, 4.1, 4)]
DOOR_WALLS = [
    (1.95, -4, 2.05, -0.7),
    (1.95, 0.7, 2.05, 4),
]  # wall at x = 2 with a 1.4 m doorway at y = 0
NO_DOTS = np.zeros((3, 4))


def value(features, name):
    return float(features[pp.NAMES.index(name)])


class PreprocessTests(unittest.TestCase):
    def test_names_match_feature_count(self):
        f = pp.Preprocessor().step(0.0, cast((0, 0), 0, ROOM), (0, 0, 0), NO_DOTS)
        self.assertEqual(len(f), len(pp.NAMES))
        self.assertTrue(np.isfinite(f).all())

    def test_nothing_in_range(self):
        f = pp.Preprocessor().step(0.0, np.full(36, 4.0), (0, 0, 0), NO_DOTS)
        self.assertEqual(
            [value(f, n) for n in ("wall_seen", "opening_seen", "mover_seen", "object_seen")],
            [0, 0, 0, 0],
        )

    def test_finds_the_wall_ahead(self):
        f = pp.Preprocessor().step(0.0, cast((0, 0), 0, [(2.0, -4, 2.1, 4)]), (0, 0, 0), NO_DOTS)
        self.assertEqual(value(f, "wall_seen"), 1)
        self.assertAlmostEqual(value(f, "wall_distance"), 2.0, delta=0.05)
        self.assertAlmostEqual(value(f, "wall_cos"), 1.0, delta=0.05)  # straight ahead
        self.assertAlmostEqual(
            value(f, "wall_dir_cos2"), -1.0, delta=0.05
        )  # the wall runs across the robot's path

    def test_finds_the_doorway_and_the_blocked_dot(self):
        dots = NO_DOTS.copy()
        dots[0] = [3.5, 2.0, 0.5, 1]  # behind the wall, up and to the left
        for robot, yaw in [((0.0, 0.0), 0.0), ((0.3, -0.5), 0.4), ((-0.5, 0.8), -0.3)]:
            with self.subTest(robot=robot, yaw=yaw):
                f = pp.Preprocessor().step(
                    0.0, cast(robot, yaw, ROOM + DOOR_WALLS), (*robot, yaw), dots
                )
                self.assertEqual(value(f, "opening_seen"), 1)
                d, si, co = (
                    value(f, "opening_distance"),
                    value(f, "opening_sin"),
                    value(f, "opening_cos"),
                )
                c, s = np.cos(yaw), np.sin(yaw)
                world = np.array(robot) + d * np.array([c * co - s * si, s * co + c * si])
                self.assertLess(np.linalg.norm(world - (2.0, 0.0)), 0.4, world)
                self.assertEqual(value(f, "dot0_blocked"), 1)

    def test_clear_line_to_dot(self):
        dots = NO_DOTS.copy()
        dots[0] = [1.5, 0.0, 0.5, 1]
        f = pp.Preprocessor().step(0.0, cast((0, 0), 0, ROOM + DOOR_WALLS), (0, 0, 0), dots)
        self.assertEqual(value(f, "dot0_blocked"), 0)

    def test_tracks_a_moving_cylinder_and_not_a_still_one(self):
        pre, mover_seen = pp.Preprocessor(), []
        for i in range(150):  # 1.5 s at 100 Hz, robot turning slowly in place
            t, yaw = i / 100, 0.3 * i / 100
            mover = (2.0, -1.0 + 0.2 * t, 0.25)  # 0.2 m/s along +y
            f = pre.step(
                t, cast((0, 0), yaw, ROOM, [mover, (-1.5, 1.5, 0.25)]), (0, 0, yaw), NO_DOTS
            )
            mover_seen.append(value(f, "mover_seen"))
        self.assertEqual(mover_seen[-1], 1)
        self.assertEqual(value(f, "object_seen"), 1)  # the still cylinder
        c, s = np.cos(yaw), np.sin(yaw)
        v = np.array(
            [
                c * value(f, "mover_v_forward") - s * value(f, "mover_v_left"),
                s * value(f, "mover_v_forward") + c * value(f, "mover_v_left"),
            ]
        )
        self.assertLess(np.linalg.norm(v - (0, 0.2)), 0.06, v)

    def test_memory_resets(self):
        pre = pp.Preprocessor()
        pre.step(0.0, cast((0, 0), 0, ROOM, [(2, 0, 0.25)]), (0, 0, 0), NO_DOTS)
        self.assertTrue(pre.tracks)
        pre.reset()
        self.assertFalse(pre.tracks)


if __name__ == "__main__":
    unittest.main()
