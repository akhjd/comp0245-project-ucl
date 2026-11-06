"""Tests for the 6 Nov row cache in train_base_torch.load: the same arrays with or without it,
at every stride."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from comp0245.training import train_base_torch as tb
from tests.test_groups import fake_dataset


class RowCacheTests(unittest.TestCase):
    def setUp(self):
        self.stride, self.cache = tb.STRIDE, tb.ROW_CACHE

    def tearDown(self):
        tb.set_stride(self.stride)
        tb.ROW_CACHE = self.cache

    def test_same_arrays_as_the_full_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "data"
            fake_dataset(
                d,
                [("ova_000", "train"), ("ova_001", "train"), ("ovb_000", "validation")],
                rows=1200,
            )
            for ms in (10, 50):
                tb.set_stride(ms)
                tb.ROW_CACHE = False
                full = tb.load("train", d, scans=3, target_ray_=True)
                tb.ROW_CACHE = True
                built = tb.load("train", d, scans=3, target_ray_=True)  # builds the caches
                read = tb.load("train", d, scans=3, target_ray_=True)  # reads them
                for a, b, c in zip(full, built, read, strict=False):
                    np.testing.assert_array_equal(a, b)
                    np.testing.assert_array_equal(a, c)
                self.assertTrue((d / "ova_000" / f"rows_{ms}ms.npz").exists())
                self.assertFalse(
                    (d / "ovb_000" / f"rows_{ms}ms.npz").exists()
                )  # only the split that was loaded
            self.assertEqual(len(list(d.rglob("*.part.npz"))), 0)


if __name__ == "__main__":
    unittest.main()
