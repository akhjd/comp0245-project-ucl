"""Tests for the mixture output (--mixture K): two different expert answers must not be averaged."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from eval_obstacles import load_model, LearnedBase, STICK
from train_base_torch import build_net, input_names, mixture_nll, VELOCITY


class MixtureTests(unittest.TestCase):
    def test_two_answers_are_not_averaged(self):
        """Same input, the expert answers +1 half the time and -1 the other half (left or right round an obstacle).
        Squared error learns 0 (straight on); the mixture keeps both and drives one of them."""
        torch.manual_seed(0)
        x = torch.zeros(400, 42)
        y = torch.ones(400, 4)
        y[200:] = -1.0
        net = build_net(42, 16, mixture=2)
        optimiser = torch.optim.Adam(net.parameters(), lr=1e-2)
        for _ in range(400):
            loss = mixture_nll(*net.candidates(net.features(x)), y).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            log_p, mean, _ = net.candidates(net.features(x[:1]))
            driven = net(x[:1])[0]
        np.testing.assert_allclose(sorted(mean[0, :, 0].tolist()), [-1.0, 1.0], atol=0.05)   # one candidate per answer
        np.testing.assert_allclose(log_p.exp()[0].tolist(), [0.5, 0.5], atol=0.05)
        self.assertGreater(driven.abs().min().item(), 0.9)                                   # not the average, 0

    def test_plain_and_old_models_unchanged(self):
        self.assertIsInstance(build_net(42, 64), torch.nn.Sequential)
        self.assertEqual(build_net(42, 64, mixture=3)(torch.zeros(5, 42)).shape, (5, 4))

    def test_sticks_to_its_choice(self):
        """Keeps the candidate it follows until another is at least STICK times as likely."""
        policy = LearnedBase.__new__(LearnedBase)
        policy.choice = None
        mean = np.array([[1.0]*4, [-1.0]*4])
        self.assertEqual(policy.pick(np.log([0.6, 0.4]), mean)[0], 1.0)
        self.assertEqual(policy.pick(np.log([0.45, 0.55]), mean)[0], 1.0)      # slightly behind: keep going
        p = 1/(1 + STICK) - 0.01
        self.assertEqual(policy.pick(np.log([p, 1 - p]), mean)[0], -1.0)       # clearly behind: switch

    def test_saved_model_loads(self):
        names = input_names(3)
        columns = [i for i, n in enumerate(names) if n not in VELOCITY]
        n = len(columns)
        net = build_net(n, 16, 3, "conv", mixture=3)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"model.pt"
            torch.save({"state_dict": net.state_dict(), "hidden": 16, "scans": 3, "encoder": "conv", "threat": False,
                        "mixture": 3, "input_names": [names[i] for i in columns], "columns": columns,
                        "x_mean": np.zeros(n), "x_std": np.ones(n), "y_mean": np.zeros(4), "y_std": np.ones(4)}, path)
            loaded, _ = load_model(path)
        x = torch.rand(5, n)
        with torch.no_grad():
            torch.testing.assert_close(loaded(x), net(x))


if __name__ == "__main__":
    unittest.main()
