# Final model weights

Small enough to keep in Git (about 30 kB each). `demo.py` uses these by default.

| File | What | Copied from (29 Oct 2026) | sha256 |
|---|---|---|---|
| `base_final.pt` | base policy: 4 wheel speeds from 42 inputs (6 target-in-robot-frame features + 36 lidar ranges; no velocity inputs), MLP 64-64; noise-robust DAgger with hard layouts, round 3 | `runs/dagger-novnoise-hard/r3/model.pt` | `2d917b39e0a214a337ad2baa28947649c99702351f8698c4eb173f502e338b89` |
| `arm_final.pt` | arm policy: 7 torques = gravity (pinocchio) + MLP 64-64 on 24 arm inputs; arm DAgger with the learned base parking, round 3 | `runs/dagger-arm-b/r3/model.pt` | `4a38c9bc51b216e0431dfefa1c050eff7654c8c8fd01cbe110ed15317167c933` |

Each file is a `torch.save` dictionary: `state_dict`, `hidden`, `input_names` and the input/output scaling
(`x_mean`, `x_std`, `y_mean`, `y_std`); the base also stores `columns`, which of the 45 base inputs it uses. Load with `eval_obstacles.load_model` and `eval_arm.load_arm`, which
check the input names. Loading needs `torch.load(..., weights_only=False)`: only load files you trust.
