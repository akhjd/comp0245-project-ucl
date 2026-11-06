import unittest

import numpy as np

from comp0245.models import base_inputs
from comp0245.sim.obstacle_env import lidar_angles
from comp0245.training.train_base_torch import (
    MIRROR_THREAT,
    MIRROR_WHEELS,
    THREAT_NAMES,
    VELOCITY,
    input_names,
    mirror_map,
    mirror_symmetric,
    speed_turn,
)

RAYS, SCANS = 72, 3


def reflect(p, base, yaw):
    """Mirror a world point across the robot's heading line (the robot itself is unchanged)."""
    u = np.array([np.cos(yaw), np.sin(yaw)])
    d = np.asarray(p[:2]) - base[:2]
    return np.concatenate([base[:2] + 2 * (d @ u) * u - d, p[2:]])


def scan(base, yaw, discs, rays=RAYS, far=6.0):
    """Lidar ranges from the base to a set of discs (centre xy, radius), straight-line maths."""
    out = np.full(rays, far)
    for k, a in enumerate(lidar_angles(rays)):
        d = np.array([np.cos(yaw + a), np.sin(yaw + a)])
        for c, r in discs:
            m = np.asarray(c) - base[:2]
            along = m @ d
            gap = r**2 - (m @ m - along**2)
            if gap >= 0 and along - np.sqrt(gap) > 0:
                out[k] = min(out[k], along - np.sqrt(gap))
    return out


def row(base, yaw, target, v, yaw_rate, scans_discs):
    obs = np.zeros(28)
    obs[0:3], obs[3], obs[4], obs[5:8] = base, np.sin(yaw), np.cos(yaw), target
    obs[22:24], obs[27] = v, yaw_rate
    return np.concatenate([base_inputs(obs)] + [scan(base, yaw, d) for d in scans_discs])


class MirrorTest(unittest.TestCase):
    def setUp(self):
        self.names = input_names(SCANS, RAYS)
        self.perm, self.sign = mirror_map(self.names)

    def test_flip_twice_is_identity(self):
        np.testing.assert_array_equal(self.perm[self.perm], np.arange(len(self.names)))
        np.testing.assert_array_equal(self.sign * self.sign[self.perm], np.ones(len(self.names)))

    def test_flipped_row_is_the_mirror_world(self):
        base, yaw = np.array([1.2, -0.4, 0.1]), 0.7
        target, v, yaw_rate = np.array([3.0, 1.1, 0.6]), np.array([0.15, 0.05]), 0.3
        discs = [
            [((2.0, 0.5), 0.25), ((0.5, 1.5), 0.2)],
            [((2.1, 0.4), 0.25)],
            [((2.2, 0.3), 0.25), ((-1.0, -1.0), 0.3)],
        ]
        x = row(base, yaw, target, v, yaw_rate, discs)
        u = np.array([np.cos(yaw), np.sin(yaw)])
        mirror_v = 2 * (v @ u) * u - v
        mirror_discs = [[(reflect(np.array(c), base, yaw), r) for c, r in d] for d in discs]
        x_mirror = row(base, yaw, reflect(target, base, yaw), mirror_v, -yaw_rate, mirror_discs)
        np.testing.assert_allclose(x[self.perm] * self.sign, x_mirror, atol=1e-9)
        # the same with the velocity inputs left out (as the room models are trained)
        cols = [i for i, n in enumerate(self.names) if n not in VELOCITY]
        perm, sign = mirror_map([self.names[i] for i in cols])
        np.testing.assert_allclose(x[cols][perm] * sign, x_mirror[cols], atol=1e-9)

    def test_wheels_swap_sides(self):
        wheels = np.array([[3.0, 1.0, 1.5, 2.5], [-1.0, 2.0, 2.0, -1.0]])  # [FR, FL, BL, BR]
        v, w = speed_turn(wheels)
        v_m, w_m = speed_turn(wheels[:, MIRROR_WHEELS])
        np.testing.assert_allclose(v_m, v)
        np.testing.assert_allclose(w_m, -w)

    def test_threat_left_parts_change_sign(self):
        expected = [-1.0 if "left" in n else 1.0 for n in THREAT_NAMES]
        np.testing.assert_array_equal(MIRROR_THREAT, expected)

    def test_symmetric_scaling_makes_the_flip_exact(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(500, len(self.names))) + rng.normal(size=len(self.names))
        mean, std = mirror_symmetric(x.mean(axis=0), x.std(axis=0), self.perm, self.sign)
        both = np.concatenate([x, x[:, self.perm] * self.sign])
        np.testing.assert_allclose(mean, both.mean(axis=0), atol=1e-5)
        np.testing.assert_allclose(std, both.std(axis=0), rtol=1e-4)
        scaled = (x - mean) / std
        np.testing.assert_allclose(
            scaled[:, self.perm] * self.sign, (x[:, self.perm] * self.sign - mean) / std, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
