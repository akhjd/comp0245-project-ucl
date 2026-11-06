"""Evade/target split (A2, 3 Nov): collision labels, the two-head model with a switch, and the driving hysteresis."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from eval_obstacles import load_model, LearnedBase, SWITCH_ON, SWITCH_OFF
from obstacle_env import ROBOT_RADIUS
from train_base_torch import build_net, collision_labels, input_names, load, train, MODE_MARGIN_M, STRIDE, VELOCITY


class Npz(dict):
    """Stands in for a loaded trajectory.npz (has .files)."""
    @property
    def files(self):
        return list(self)


def straight_run(robot_v, mover_xy, mover_v, rows=100, yaw_rate=0.0):
    """Robot from the origin at constant world velocity robot_v (and yaw rate), one mover from mover_xy at mover_v;
    rows STRIDE ms apart."""
    t = np.arange(rows)*STRIDE/1000
    obs = np.zeros((rows, 28))
    obs[:, 0:2] = np.outer(t, robot_v)
    obs[:, 4] = 1.0                                    # facing +x
    obs[:, 22:24] = robot_v
    obs[:, 27] = yaw_rate
    mover = np.asarray(mover_xy) + np.outer(t, mover_v)
    still = np.tile([0.0, 0.8], (rows, 1))             # a still obstacle close to the path: never counts
    base = np.hstack([obs[:, 0:2], np.zeros((rows, 1))])
    return Npz(observations=obs, time_s=t, base_m=base, obstacle_xy=np.stack([mover, still], axis=1))


class CollisionLabelTests(unittest.TestCase):
    def test_head_on_is_evade(self):
        """Robot +x at 0.5 m/s, mover 3 m ahead coming at 0.5 m/s: they meet within 3 s."""
        z = straight_run([0.5, 0.0], [3.0, 0.0], [-0.5, 0.0])
        labels = collision_labels(z, [True, False], [0.2, 0.3], 1)
        self.assertTrue(labels.all())

    def test_passing_behind_is_not_evade(self):
        """A mover crossing the robot's line 1.5 m behind it: closest approach about 2.3 m, never evade."""
        z = straight_run([0.5, 0.0], [-1.5, 2.0], [0.0, -0.8])
        self.assertFalse(collision_labels(z, [True, False], [0.2, 0.3], 1).any())

    def test_margin_and_still_obstacles(self):
        """Robot standing still, mover passing at a fixed side gap: evade only when the gap is below the margin."""
        limit = ROBOT_RADIUS + 0.2 + MODE_MARGIN_M
        for gap, expected in ((limit - 0.05, True), (limit + 0.05, False)):
            z = straight_run([0.0, 0.0], [1.0, -gap], [-0.5, 0.0])
            self.assertEqual(bool(collision_labels(z, [True, False], [0.2, 0.3], 1)[0]), expected)
        z = straight_run([0.5, 0.0], [3.0, 0.0], [-0.5, 0.0])
        self.assertFalse(collision_labels(z, [False, False], [0.2, 0.3], 1).any())   # nothing moves

    def test_robot_turning_is_extrapolated_on_its_arc(self):
        """Robot at 0.5 m/s turning left at 0.5 rad/s: a circle of 1 m radius round (0, 1), through (1, 1) after a
        quarter turn (3.1 s), and within the 0.97 m limit of it after about 2 s. Straight on it passes (1, 1) at
        1.0 m: not evade. (A one-row run: the 'mover' has no velocity yet.)"""
        on_arc = straight_run([0.5, 0.0], [1.0, 1.0], [0.0, 0.0], rows=1, yaw_rate=0.5)
        straight = straight_run([0.5, 0.0], [1.0, 1.0], [0.0, 0.0], rows=1)
        self.assertTrue(collision_labels(on_arc, [True, False], [0.2, 0.3], 1)[0])
        self.assertFalse(collision_labels(straight, [True, False], [0.2, 0.3], 1)[0])


def fake_dataset(folder, rays=36, episodes=2, rows=300, split="train", expert_evade=False):
    """A tiny dataset folder like collect_obstacles.py writes: half the rows of each run have a mover coming head-on,
    and the expert turns (wheels +1/-1) only then."""
    folder.mkdir()
    manifest = {"lidar_rays": rays, "saved_every": STRIDE, "episodes": []}   # rows already STRIDE ms apart
    rng = np.random.default_rng(len(str(folder)))
    for k in range(episodes):
        z = straight_run([0.5, 0.0], [2.0, 0.0] if k % 2 == 0 else [2.0, 5.0], [-0.5, 0.0], rows=rows)
        ep = folder/f"ep{k}"
        ep.mkdir()
        lidar = rng.uniform(0.5, 4.0, (len(z["observations"]), rays))
        actions = np.zeros((len(lidar), 11))
        actions[:, :4] = [1.0, -1.0, -1.0, 1.0] if k % 2 == 0 else [2.0, 2.0, 2.0, 2.0]
        extra = {"expert_evade": np.full(len(lidar), k % 2 == 0)} if expert_evade else {}
        np.savez(ep/"trajectory.npz", lidar=lidar, actions=actions, **z, **extra)
        (ep/"episode.json").write_text(json.dumps({"scenario": {"obstacles": [{"r": 0.2}, {"r": 0.3}]}}))
        manifest["episodes"].append({"path": f"ep{k}", "split": split, "success": True,
                                     "obstacle_is_moving": [True, False]})
    (folder/"manifest.json").write_text(json.dumps(manifest))
    return folder


class ModesModelTests(unittest.TestCase):
    def test_load_gives_labels_and_expert_source(self):
        with tempfile.TemporaryDirectory() as d:
            folder = fake_dataset(Path(d)/"data")
            x, y, hard, m = load("train", folder, mode_label="collision")
            self.assertEqual(len(m), len(x))
            self.assertTrue(0.3 < m.mean() < 0.7)           # the head-on run is evade, the far one is not
            with self.assertRaisesRegex(KeyError, "expert_evade"):
                load("train", folder, mode_label="expert")
            folder = fake_dataset(Path(d)/"labelled", expert_evade=True)
            self.assertAlmostEqual(load("train", folder, mode_label="expert")[3].mean(), 0.5)

    def test_trains_saves_and_loads(self):
        """2 epochs on the fake data (72 rays, conv, threat too): model.pt has rays and modes and loads back."""
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as d:
            data = fake_dataset(Path(d)/"data", rays=72)
            val = fake_dataset(Path(d)/"val", rays=72, split="validation")
            train([data], out=Path(d)/"model", epochs=2, validation=val, no_velocity=True, scans=3, hidden=16,
                  encoder="conv", threat=True, modes=True)
            model, saved = load_model(Path(d)/"model"/"model.pt")
            info = json.loads((Path(d)/"model"/"training.json").read_text())
        self.assertTrue(saved["modes"])
        self.assertEqual(saved["rays"], 72)
        self.assertTrue(model.modes)
        self.assertEqual(len(saved["input_names"]), len(input_names(3, 72)) - len(VELOCITY))
        self.assertIn("evade_share_validation", info)
        logit, both = model.heads(model.features(torch.zeros(5, len(saved["input_names"]))))
        self.assertEqual((logit.shape, both.shape), ((5,), (5, 2, 4)))

    def test_modes_and_mixture_refused(self):
        with self.assertRaises(ValueError):
            build_net(42, 16, mixture=3, modes=True)

    def test_saved_modes_model_loads(self):
        names = input_names(1)
        net = build_net(len(names), 16, modes=True)
        n = len(names)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"model.pt"
            torch.save({"state_dict": net.state_dict(), "hidden": 16, "scans": 1, "encoder": "mlp", "modes": True,
                        "input_names": names, "columns": list(range(n)),
                        "x_mean": np.zeros(n), "x_std": np.ones(n), "y_mean": np.zeros(4), "y_std": np.ones(4)}, path)
            loaded, _ = load_model(path)
        x = torch.rand(6, n)
        with torch.no_grad():
            torch.testing.assert_close(loaded(x), net(x))


class HysteresisTests(unittest.TestCase):
    def test_mode_holds_inside_the_band(self):
        self.assertEqual((SWITCH_ON, SWITCH_OFF), (0.6, 0.4))
        policy = LearnedBase.__new__(LearnedBase)
        policy.evading, policy.mode_log, policy.switch_log = False, [], []
        probs = [0.3, 0.55, 0.59, 0.65, 0.5, 0.41, 0.45, 0.39, 0.5, 0.59, 0.61]
        modes = [policy.switch(p) for p in probs]
        self.assertEqual(modes, [False, False, False, True, True, True, True, False, False, False, True])
        self.assertEqual(policy.mode_log, modes)

    def test_driving_uses_the_chosen_head(self):
        """A modes network whose evade head says +1 and target head -1: the wheels follow the switch."""
        names = input_names(1)
        n = len(names)
        net = build_net(n, 8, modes=True)
        with torch.no_grad():
            for p in net.parameters():
                p.zero_()
            net.wheels.bias[:] = torch.tensor([-1.0]*4 + [1.0]*4)
        saved = {"x_mean": np.zeros(n, np.float32), "x_std": np.ones(n, np.float32), "y_mean": np.zeros(4),
                 "y_std": np.ones(4)}
        policy = LearnedBase(net, saved)
        state = {"x": np.zeros(28), "lidar": np.full(36, 4.0), "q": np.zeros(11), "qd": np.zeros(11)}
        for bias, wheel in ((2.0, 1.0), (0.0, 1.0), (-2.0, -1.0)):     # p = 0.88, 0.5 (stays evade), 0.12
            with torch.no_grad():
                net.switch.bias[:] = bias
            self.assertEqual(policy.act(state)[0], wheel)
        with tempfile.TemporaryDirectory() as d:
            policy.save_modes(d)
            saved_modes = np.load(Path(d)/"modes.npz")
            np.testing.assert_array_equal(saved_modes["evade"], [True, True, False])


if __name__ == "__main__":
    unittest.main()
