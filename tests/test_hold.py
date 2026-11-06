import unittest
import numpy as np
import torch
from train_base_torch import build_net
from eval_obstacles import LearnedBase, HOLD_ON
from models import base_inputs


class HoldTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = build_net(9 + 72*3, 64, 3, "conv", True, rays=72, hold=True).eval()
        n = 9 + 72*3
        self.saved = {"x_mean": np.zeros(n, np.float32), "x_std": np.ones(n, np.float32),
                      "y_mean": np.full(4, 0.5), "y_std": np.ones(4), "scans": 3, "rays": 72}
        self.state = {"q": np.zeros(11), "qd": np.zeros(11), "x": np.zeros(30), "lidar": np.full(72, 4.0)}

    def set_hold(self, bias):
        with torch.no_grad():
            self.model.hold[2].weight.zero_()
            self.model.hold[2].bias.fill_(bias)

    def test_hold_drives_exactly_zero(self):
        self.set_hold(5.0)
        policy = LearnedBase(self.model, self.saved)
        self.assertTrue(np.all(policy.act(self.state)[:4] == 0))
        self.assertGreater(policy.hold_log[-1], HOLD_ON)

    def test_no_hold_drives_the_regression(self):
        self.set_hold(-5.0)
        policy = LearnedBase(self.model, self.saved)
        with torch.no_grad():
            x = torch.from_numpy(np.hstack([base_inputs(self.state["x"]), np.full(216, 4.0)]).astype(np.float32))[None]
            want = self.model.wheels(self.model.features(x))[0].numpy() + 0.5
        np.testing.assert_allclose(policy.act(self.state)[:4], want, rtol=1e-5)

    def test_latch_keeps_holding_until_low(self):
        policy = LearnedBase(self.model, {**self.saved, "hold_latch": True})
        for _ in range(500):
            self.assertTrue(policy.holding(0.9))
        self.assertTrue(policy.holding(0.3))        # latched: a dip above HOLD_OFF still holds
        self.assertFalse(policy.holding(0.1))       # below HOLD_OFF: free again
        plain = LearnedBase(self.model, self.saved)
        for _ in range(500):
            plain.holding(0.9)
        self.assertFalse(plain.holding(0.3))        # not latched: only above HOLD_ON

    def test_forward_unchanged_by_hold(self):
        x = torch.randn(5, 9 + 216)
        plain = build_net(9 + 216, 64, 3, "conv", True, rays=72)
        plain.load_state_dict({k: v for k, v in self.model.state_dict().items() if not k.startswith("hold.")})
        with torch.no_grad():
            torch.testing.assert_close(plain(x), self.model(x))


if __name__ == "__main__":
    unittest.main()
