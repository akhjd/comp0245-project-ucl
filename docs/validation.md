# Snapshot validation

29 October 2026: Python compilation passed and all 15 existing unit tests passed using the original RoboEnv Python 3.14 environment. The small synthetic MLP test emitted expected batch-size and convergence warnings. No new dataset collection, full training, GUI rollout or fresh-machine installation was performed for repository preparation.

The first test invocation from the parent directory failed because project modules were not on the import path; running from the project root as documented passed.

29 October 2026 (evening update): all 15 unit tests pass; every module compiles; `comp0245/apps/demo.py full` and `comp0245/apps/demo.py base` were run headless from a clean clone with the bundled weights (validationd_000: full task success in 22.5 s, base parked within 3.2 cm and 0.025 rad, no contact). Fresh-machine installation is still not validated.
