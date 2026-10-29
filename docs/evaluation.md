# Evaluation and reproducibility

These are project recommendations, not imposed numerical assignment criteria.

1. Freeze a scenario manifest separating training, validation and untouched test episodes by targets/initial conditions. Use validation for model selection. If test results influence development, create a new held-out test set.
2. Compare an expert reference and a simple regressor with candidate learned models under the same simulator settings. Report offline action errors separately from closed-loop end-effector error and task success.
3. Report all evaluation episodes, including timeouts/failures. Record the implemented success tolerance, dwell time and time budget. Examine generalisation to held-out targets/starts, chosen adversities, collision behaviour where measured, and inference timing/parameter count.
4. Save exact commands, configuration, seed, dataset/manifest hashes, code commit and dependency versions with each run. Freeze fitted preprocessing with the model. Repeat across seeds when practical and disclose sample counts and uncertainty.
5. Record CPU/GPU, OS, Python version, simulator step/action rate, external repository revisions and all local modifications. Seed control does not guarantee bitwise reproducibility across hardware.

`docs/dependencies.md` records external checkout revisions at snapshot time. Both checkouts had local modifications: revisions alone do not reproduce the environment. Preserve/review those changes before claiming a fresh clone reproduces results.

Known limitations: legacy documentation predates several experimental changes; arm_data.py contains a workstation-specific URDF path; experimental evaluators may contain run-specific defaults. Update these before moving machines. No final model performance is claimed by this repository setup.
