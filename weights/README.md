# Final model weights

Small enough to keep in Git (30-60 kB each). `demo.py` uses these by default.

| File | What | Copied from (29 Oct 2026) | sha256 |
|---|---|---|---|
| `base_final.pt` | base policy: 4 wheel speeds from 42 inputs (6 target-in-robot-frame features + 36 lidar ranges; no velocity inputs), MLP 64-64; noise-robust DAgger with hard layouts, round 3 | `runs/dagger-novnoise-hard/r3/model.pt` | `2d917b39e0a214a337ad2baa28947649c99702351f8698c4eb173f502e338b89` |
| `arm_final.pt` | arm policy: 7 torques = gravity (pinocchio) + MLP 64-64 on 24 arm inputs; arm DAgger with the learned base parking, round 3 | `runs/dagger-arm-b/r3/model.pt` | `4a38c9bc51b216e0431dfefa1c050eff7654c8c8fd01cbe110ed15317167c933` |
| `arm_rooms.pt` | arm for the room robot (added 5 Nov; same layout as `arm_final.pt`): retrained with 1,000 expert reaches from where the room base parks, then 2 rounds of arm DAgger on parked spots. 49 validation parking spots 45 (arm_final 25), 200 held-out synthetic spots 188, full robot 37/60 validation rooms. The arm unfolds first (eval_arm.py --arm-fold). The final test (45/80) was run with `arm_final.pt` | `runs/arm-parked/dagger-r2/arm/model.pt` | `2b473dbbe3097614662b6778fc4f29580c80b0f0f159203e11d19a183b22ddd2` |
| `base_rooms.pt` | room base (added 6 Nov): 4 wheel speeds from 3 lidar scans (72 rays, 0.5 s apart) + target inputs, no velocity; learned ray filter (conv) + threat head (training only) + learned hold output with latch; window-teacher DAgger with crossing movers, round 3. 60 validation rooms: 37 parked, 12 with a mover hit; final test 48/80 parked. Used by `run_scene.py` | `runs/crossing-hold/r3-hold-latch/model.pt` | `e882f0b11a51c625018b10d8fd6a795e55f656e5783d50ba799258526eaf1f5d` |

Each file is a `torch.save` dictionary: `state_dict`, `hidden`, `input_names` and the input/output scaling
(`x_mean`, `x_std`, `y_mean`, `y_std`); the base also stores `columns`, which of the 45 base inputs it uses. Load with `eval_obstacles.load_model` and `eval_arm.load_arm`, which
check the input names. Loading needs `torch.load(..., weights_only=False)`: only load files you trust.
