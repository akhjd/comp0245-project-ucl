"""Tests for the wheels MLP's training options (train_mlp.py): --dropout, --weight-decay, --evals-per-epoch, and DAgger reusing them.

    python -m unittest tests.test_mlp_training      run from the project folder; needs runs/dataset

Uses one thread and a few runs, so it can run beside a long job.
"""
import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
import train_mlp as t

ROOT = Path(__file__).resolve().parent.parent
DATA = str(ROOT/"runs/dataset")


def settings(**changes):
    base = dict(no_velocity=True, no_xy=False, dot_polar=True, lidar_features=False, seed=0, mirror=False, hidden=32, layers=2, lr=1e-3,
                lr_patience=4, resume=False, epochs=2, batch=1024, patience=50, lidar_noise=0.0, velocity_noise=0.0, yaw_rate_noise=0.0,
                dropout=0.0, weight_decay=0.0, evals_per_epoch=1)
    return argparse.Namespace(**{**base, **changes})


class ModelTests(unittest.TestCase):
    def test_dropout_layers_only_when_asked(self):
        self.assertFalse(any(isinstance(m, nn.Dropout) for m in t.make_model(32, 2, 10)))
        self.assertEqual(sum(isinstance(m, nn.Dropout) for m in t.make_model(32, 2, 10, 0.2)), 2)


@unittest.skipUnless(Path(DATA).exists(), "needs runs/dataset")
class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.train = t.load(DATA, "train", limit=8)
        cls.val = t.load(DATA, "validation", limit=4)

    def run_fit(self, tmp, **changes):
        out = Path(tmp)/"m"
        out.mkdir()
        args = settings(**changes)
        t.fit(self.train, self.val, out, args)
        with (out/"metrics.csv").open() as f:
            rows = list(csv.DictReader(f))
        return out, rows, args

    def test_evals_per_epoch_gives_fractional_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, rows, _ = self.run_fit(tmp, evals_per_epoch=4)
            self.assertEqual([float(r["epoch"]) for r in rows], [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
            result = json.loads((out/"result.json").read_text())
            self.assertEqual((result["epochs_run"], result["checks_run"]), (2.0, 8))

    def test_one_eval_per_epoch_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, rows, _ = self.run_fit(tmp)
            self.assertEqual([r["epoch"] for r in rows], ["1", "2"])

    def test_regularised_model_saves_loads_and_predicts_the_same(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, _, _ = self.run_fit(tmp, dropout=0.2, weight_decay=0.01, evals_per_epoch=2)
            model, saved = t.load_policy(out/"best.pt")
            self.assertEqual(saved["training"], {"dropout": 0.2, "weight_decay": 0.01, "evals_per_epoch": 2, "lr": 1e-3})
            self.assertFalse(model.training)        # dropout is off when driving
            scale, idx = t.Scale(saved=saved), np.arange(0, len(self.val.wheels), 100)
            self.assertTrue(np.array_equal(t.predict(model, scale, self.val, idx), t.predict(model, scale, self.val, idx)))

    def test_weight_decay_shrinks_weights(self):
        norms = {}
        for wd in (0.0, 0.5):
            with tempfile.TemporaryDirectory() as tmp:
                out, _, _ = self.run_fit(tmp, weight_decay=wd, epochs=3)
                last = torch.load(out/"last.pt", weights_only=False)["model"]
                norms[wd] = sum(float(v.norm()) for k, v in last.items() if k.endswith("weight"))
        self.assertLess(norms[0.5], norms[0.0])

    def test_dagger_reuses_training_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, _, _ = self.run_fit(tmp, dropout=0.2, weight_decay=0.01, evals_per_epoch=2, epochs=1)
            _, saved = t.load_policy(out/"best.pt")
            flags = [f"--{k.replace('_', '-')}={v}" for k, v in saved["training"].items()]   # as in dagger_mlp.py
            parsed = settings()
            for flag in flags:
                key, value = flag[2:].split("=")
                setattr(parsed, key.replace("-", "_"), type(getattr(parsed, key.replace("-", "_")))(value))
            self.assertEqual((parsed.dropout, parsed.weight_decay, parsed.evals_per_epoch, parsed.lr), (0.2, 0.01, 2, 1e-3))

    @unittest.skipUnless((ROOT/"runs/mlp3/best.pt").exists(), "needs runs/mlp3")
    def test_old_model_still_loads_without_dropout(self):
        model, saved = t.load_policy(ROOT/"runs/mlp3/best.pt")
        self.assertNotIn("training", saved)
        self.assertFalse(any(isinstance(m, nn.Dropout) for m in model))


if __name__ == "__main__":
    unittest.main()
