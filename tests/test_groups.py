"""Tests for train_base_torch's balance_groups and set_stride (5 Nov), on a tiny made-up dataset."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import train_base_torch as tb


def fake_dataset(folder, episodes, rows=200, rays=72):
    """episodes: list of (scenario_id, split). Saved every 1 ms, as collect_obstacles.py saves."""
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    eps = []
    for sid, split in episodes:
        (folder/sid).mkdir()
        obs = np.zeros((rows, 28))
        obs[:, 4], obs[:, 5:8] = 1.0, [2.0, 0.5, 0.6]
        np.savez(folder/sid/"trajectory.npz", observations=obs, lidar=rng.uniform(0.5, 4, (rows, rays)),
                 actions=rng.normal(size=(rows, 11)), time_s=np.arange(rows)/1000)
        eps.append({"path": sid, "scenario_id": sid, "split": split, "success": True})
    (folder/"manifest.json").write_text(json.dumps({"lidar_rays": rays, "episodes": eps}))


class GroupTests(unittest.TestCase):
    def test_group_of(self):
        self.assertEqual(tb.group_of("ovc3_0012"), "ovc3")
        self.assertEqual(tb.group_of("trainrmv2_013"), "trainrmv2")

    def test_load_reports_groups_and_stride(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)/"data"
            fake_dataset(d, [("ova_000", "train"), ("ovb_000", "train"), ("ovb_001", "train")])
            groups = []
            try:
                tb.set_stride(10)
                x, *_ = tb.load("train", d, groups=groups)
                self.assertEqual(len(x), 3*20)                       # 200 rows at 1 ms, every 10th
                self.assertEqual(sorted(set(np.concatenate(groups))), ["ova", "ovb"])
                tb.set_stride(20)
                x, *_ = tb.load("train", d)
                self.assertEqual(len(x), 3*10)
            finally:
                tb.set_stride(10)
            with self.assertRaises(ValueError):
                tb.set_stride(15)

    def test_balance_groups_weights_each_group_equally(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)/"data"
            fake_dataset(d, [("ova_000", "train"), ("ovb_000", "train"), ("ovb_001", "train"), ("ovb_002", "train"),
                             ("ova_100", "validation")])
            tb.train([str(d)], [], Path(tmp)/"out", True, epochs=1, validation=str(d), scans=3, hidden=16,
                     encoder="conv", balance_groups=True)
            info = json.loads((Path(tmp)/"out"/"training.json").read_text())
            self.assertEqual(info["group_rows"], {"ova": 20, "ovb": 60})
            self.assertTrue(info["balance_groups"])


if __name__ == "__main__":
    unittest.main()
