"""Tests for leakage, time alignment conventions and the policy interchange contract."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from models import History, stack_episode, clip_action, make_model, FEATURE_NAMES, target_in_base_frame, base_features, base_inputs, BASE_INPUT_NAMES
from pipeline import read_json, ROOT, scenarios, load_split

class PipelineTests(unittest.TestCase):
    def test_history_is_causal_and_resets(self):
        x=np.array([[1,2],[3,4],[5,6]],dtype=float)
        np.testing.assert_equal(stack_episode(x,2),[[1,2,1,2],[1,2,3,4],[3,4,5,6]])
        np.testing.assert_equal(History(2).append(np.array([9,8])),[9,8,9,8])

    def test_saturation_units_and_nonfinite(self):
        config=read_json(ROOT/"settings.json")
        np.testing.assert_equal(clip_action(np.full(11,100),config),[12]*4+[30]*7)
        with self.assertRaises(ValueError):clip_action(np.full(11,np.nan),config)

    def test_models_share_interface_and_train_only_scaling(self):
        config=read_json(ROOT/"settings.json")
        config["models"]["mlp"].update(hidden_layer_sizes=[8],max_iter=10)
        rng=np.random.default_rng(7)
        x=rng.normal(size=(60,len(FEATURE_NAMES)));y=rng.normal(size=(60,11))
        for name in ["ridge","mlp","forest"]:
            model=make_model(name,config);model.fit(x,y)
            self.assertEqual(model.predict(x[:3]).shape,(3,11))
            np.testing.assert_allclose(model.regressor_.steps[0][1].mean_,x.mean(axis=0))
            before=model.regressor_.steps[0][1].mean_.copy()
            model.predict(x[:3]+1000)
            np.testing.assert_equal(model.regressor_.steps[0][1].mean_,before)

    def test_scenarios_have_disjoint_targets(self):
        s=scenarios(ROOT/"scenarios.json")
        self.assertEqual(set(x["split"] for x in s),{"train","validation","test"})

    def test_episode_split_excludes_test_and_failed_demonstrations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            episodes=[]
            for name,split,success,value in [("good","train",True,1),("heldout","test",True,999),
                                              ("failed","train",False,777)]:
                (root/name).mkdir()
                np.savez(root/name/"trajectory.npz",observations=np.full((3,len(FEATURE_NAMES)),value),
                         actions=np.full((3,11),value))
                episodes.append({"path":name,"scenario_id":name,"split":split,"success":success})
            x,y,ids=load_split(root,{"episodes":episodes},"train",3)
            self.assertEqual(ids,["good"])
            self.assertTrue((x==1).all() and (y==1).all())
    
    def test_target_in_base_frame(self):
        base0 = np.array([0., 0., 0.])
        # facing +x, target 1 m ahead
        np.testing.assert_allclose(target_in_base_frame(base0, 0.0, np.array([1, 0, 0.5])), [1, 0, 0.5], atol=1e-12)
        # facing +y (yaw = pi/2), target at world (0, 1): where is it from the robot?
        np.testing.assert_allclose(target_in_base_frame(base0, np.pi/2, np.array([0, 1, 0.5])), [1, 0, 0.5], atol=1e-12)
        # facing +y, target at world (1, 0): is it on the robot's left or right?
        np.testing.assert_allclose(target_in_base_frame(base0, np.pi/2, np.array([1, 0, 0.5])), [0, -1, 0.5], atol=1e-12)
        # robot at (2, 3) facing +x, target at (3, 3)
        np.testing.assert_allclose(target_in_base_frame(np.array([2., 3, 0]), 0.0, np.array([3, 3, 0.5])), [1, 0, 0.5], atol=1e-12)
        # target straight ahead -> heading error 0 -> sin 0, cos 1
        np.testing.assert_allclose(base_features(base0, 0.0, np.array([2, 0, 0.5]))[3:], [0, 1], atol=1e-12)

    def test_base_inputs(self):
        obs = np.zeros(len(FEATURE_NAMES))
        obs[3], obs[4] = 1.0, 0.0          # facing +y (sin 1, cos 0)
        obs[5:8] = [0, 2, 0.5]             # target 2 m north
        obs[22] = 0.3                      # moving east at 0.3 m/s
        obs[27] = 0.1                      # turning left slowly
        x = base_inputs(obs)
        self.assertEqual(len(x), len(BASE_INPUT_NAMES))
        np.testing.assert_allclose(x, [2, 0, 2, 0, 1, 0.5, 0, -0.3, 0.1], atol=1e-12)

if __name__=="__main__":unittest.main()
