# Design choices mapped to COMP0245 course material

For the report's "justify your choices" (the brief: *Methodology*, *Justify your choices*). Each row: what we
did, the course material it comes from, and what to cite. Course files are in `~/Documents/Subject Files/`.
External papers are marked; check exact titles and years before citing (written from memory).

Course slide sets used below:
- **W1-1** `week1 (1).pdf`: foundational concepts (prediction vs inference, bias-variance, learning paradigms)
- **W1-2** `week_1_2 (1).pdf`: regression basics (least squares, regularisation, cross-validation)
- **W2-1** `week2 (1).pdf`: beyond linearity (polynomials, KNN, trees, GPs, neural networks)
- **TREES** `DT_RF_bagging_boosting.pdf`: decision tree / random forest / boosting lab
- **W2-2** `Unsupervised Methods (2).pdf`: PCA, clustering
- **W3-1** `Active Learning_with_biblio.pdf`: active learning, Bayesian optimisation (incl. learning PID gains)
- **L05** `Lecture 05.pdf`: introduction to deep learning (MLPs, ReLU, loss, training, PyTorch)
- **PREP** `Data Preprocessing for Robotics.pdf`: clean, scale, engineer, balance, format
- **K-2a / K-3a** COMP0246 kinematics lectures (forward kinematics, inverse kinematics / workspace)

---

## 1. Problem formulation and data

| Choice | Course link | Report wording / citation |
|---|---|---|
| Behaviour cloning as supervised regression: (state, target) -> (wheel speeds, torques) | Assignment brief; W1-1 (supervised learning: predictors X -> response Y); L05 "Imitation learning requires highly non-linear modelling, smooth action functions, generalisable to slightly different configurations" | Cite the brief and L05; external: Pomerleau 1989 (ALVINN) as the origin of behaviour cloning |
| Many random targets and start poses instead of a few fixed ones | W1-1 / W1-2 (generalisation, variance of estimators: more varied samples); brief "generalise to new targets and/or initial conditions" | Our evidence: 4 targets failed, 60 then 220 randomised episodes |
| Separate train / validation / test scenarios; test untouched until the end | W1-2 cross-validation, W2-1 / TREES test-set error | Validation for all decisions; test once at the end |
| Keep every 10th 1 kHz row | PREP "Format" step (sensor streams at different rates; redundant samples) | Neighbouring 1 ms rows are near-duplicates; 10x less memory, same information |
| Expert quality metrics (path ratio, turning, clearance), not just success | PREP "Garbage in" / the cost of bad data | A 100%-successful but wiggly expert is a bad teacher (LESSONS.md #14) |
| Rare events matter more than the average (a single collision) | PREP module 4 "Handling rare physical events: 10,000 passes, 1 crash" and weighted losses | Report contacts and minimum clearance separately from accuracy; possible extension: weight near-obstacle rows |

## 2. Features (inputs)

| Choice | Course link | Report wording / citation |
|---|---|---|
| Robot-frame features (forward, left, distance) instead of world coordinates | PREP "Engineer" step (feature engineering for kinematics); K-2a frames and transforms | Same situation anywhere in the world looks the same to the model |
| Heading error as sin / cos, not a raw angle | PREP "The problem with angles": angle boundaries break neural continuity (they use quaternions for 3D; sin/cos is the 2D equivalent) | Avoids the jump at +-180 degrees |
| Standardise inputs and outputs (z-score) using training statistics only | PREP "Scale": magnitude dominance (velocity overwhelms position), scaling turns a narrow ravine into a round bowl so gradient descent converges | Our own failure: near-constant inputs exploded under scaling (LESSONS.md #3) |
| Lidar as an extra sensor | Brief: "Would extra sensors / input data help?"; PREP feature-level fusion of exteroceptive (lidar) and proprioceptive (pose, velocity) data by concatenation | Our 9 + 36 = 45 input vector is exactly "fusion at the feature level" |
| Dropping velocity inputs (copycat) | PREP "Injecting momentum": single frames lack motion information, so velocity/history are normally *added*. Our result shows the opposite risk: in imitation, own-velocity inputs enable a shortcut | External: de Haan et al. 2019 "Causal confusion in imitation learning"; Wen et al. 2020 "Fighting copycat agents" |
| Single instant vs a time window | Brief: "single instant vs temporal interval?"; PREP temporal context | Our early history-3 runs (2 ms window) vs history-1; proposed experiment: a longer window (e.g. 0.2 s) |

## 3. Model

| Choice | Course link | Report wording / citation |
|---|---|---|
| Neural network (MLP) instead of ridge / random forest | W2-1 list of non-linear models; TREES lab; L05 MLPs for regression | Our closed-loop baselines: ridge and MLP drifted, the forest's piecewise-constant output made the arm chatter (archive/runs) |
| ReLU activations | L05 "Activation functions: ReLU is easy and fast to compute, gradient is only 0 or 1"; "The effect of ReLU: each output is piecewise linear"; more inputs / layers give more linear regions | Widget explanation: each neuron is a hinge; training places the bends |
| No activation on the output layer | L05 "Unbounded" outputs for regression (vs bounded sigmoid) | Wheel speeds must be allowed to be negative |
| Small network (45-64-64-4, 7,364 parameters) | L05 shallow vs deep, parameter counts; brief: "How efficient is your solution (memory, parameters, computation time)?" | 0.02-0.1 ms per step vs the 1 ms control period; a 27M-parameter MLP measured 3 ms per step (too slow) |
| Separate base and arm models | Brief's diagram (one regressor) vs the expert's own structure (base phase, then arm); W1-1 model-based control limits | Justify as matching the task's structure; report it as two regressors |

## 4. Training

| Choice | Course link | Report wording / citation |
|---|---|---|
| Mean squared error loss | L05 "Loss functions: regression -> sum of squared errors"; W1-2 RSS / least squares | Same quantity as the RMSE we report |
| Adam, learning rate 1e-3 | L05 (Adam, learning rate, gradient descent); W1-2 gradient descent | External: Kingma and Ba 2015 |
| Mini-batches of 256, shuffled every epoch | L05 (stochastic / mini-batch gradient descent, batch size) | Many reliable updates per epoch; noise helps generalisation |
| Early stopping (keep best-validation epoch) | W1-1 / W1-2 overfitting and the bias-variance trade-off; L05 training and generalising | Validation loss bottomed at epoch 19 then rose |
| Learning-rate schedule (possible extension) | L05 learning rate; gradient-descent step size covered in class | Proposed one-change experiment |

## 5. Evaluation

| Choice | Course link | Report wording / citation |
|---|---|---|
| Closed-loop evaluation, not only offline RMSE | Brief: "Design an evaluation methodology"; "How robust is your solution to potential adversities?" | Headline result: offline error and closed-loop success can move in opposite directions (velocity ablation) |
| Metrics: parking-distance and facing error, contacts, clearance, reachability, time, parameters, inference time | Brief's four questions: accuracy, generalisation, robustness, efficiency | One table per question |
| Ablations (change one thing) | W1-2 / TREES hyperparameter analysis ("analyse the effect of max_depth..."); good experimental practice | Velocity on/off, data amount, DAgger rounds |
| Obstacles as "adversities" | Brief: robustness to adversities | Physics change (wheel grip 0.1) stated openly |

## 6. Expert, robot and reach zone

| Choice | Course link | Report wording / citation |
|---|---|---|
| Reach zone: arm reach as a sphere around the shoulder, sliced at the target height | K-2a forward kinematics (we measured the 0.985 m reach by FK sampling); K-3a inverse kinematics and workspace | External: Zacharias et al. 2007 (capability maps); Vahrenkamp et al. 2013 (base placement by reachability inversion) |
| 75% comfort margin | K-3a singular / stretched configurations near the workspace boundary | Verified in simulation: 24/24 reached, 6 mm hand error |
| Expert design: sensor-only, memoryless steer-around field | (beyond the course) external: Khatib 1986 potential fields | Memoryless so a student can copy it, and so DAgger works |
| Arm PD hold, PID for a learned-setpoint arm | W3-1 "Learning the PID gains for a quadrotor" (Bayesian optimisation); lab_writeup PD control | Note: the brief asks the model to output torques; see the arm proposal |

## 7. Learning-from-the-learner (DAgger)

| Choice | Course link | Report wording / citation |
|---|---|---|
| DAgger: the policy drives, the expert labels | W3-1 Active learning: the learner chooses which samples get labelled by an oracle (here the expert, a free oracle) | External: Ross, Gordon and Bagnell 2011 (DAgger); compounding error is their result. Framing for the report: DAgger is active learning where the queries are the states the policy itself visits |

---

## 8. Arm design notes

| Choice | Course link | Report wording / citation |
|---|---|---|
| Physics for the known part, learning for the rest: torque = gravity_torque(q) + network | Brief asks the model to output torques; K-2a / K-3a kinematics and dynamics | The learned torque mapping only holds for poses like those in training. Model-based inverse dynamics would handle new poses exactly, so the remaining limit has a non-ML fix. Trade-off: more built-in code, which the faculty prefer to avoid. External: residual learning; check a citation before use |
| Goal arm pose rejected (future work) | K-3a inverse kinematics; redundancy: 7 joints reaching a 3D point | Solve one good final pose and steer to it. It would respect joint limits, but a PD controller could then do the job without a network, the expert's data does not head for that pose, and our misses are not at joint limits. Evidence: the final arm's 3 failures in 40 had no joint at a limit and got within 1.2-2.1 cm |

---

## Course topics we have NOT used yet (possible quick extra experiments)

1. **Bayesian optimisation (W3-1).** Tune hyperparameters (hidden size, learning rate, data stride) or the
   arm's PD gains with BO, using closed-loop validation success as the cost. The course even has a lab script
   (`LAB_SESSION.../week3/BO_for_pid_tuning.py`). Strong link to the course, moderate effort.
2. **Cross-validation (W1-2).** We use one validation split; k-fold over scenarios would give error bars.
   Cheap offline; for closed-loop, use several training seeds instead.
3. **Weighted loss for rare events (PREP module 4).** Weight rows near obstacles or near the stop more
   heavily. One-line change in the loss; directly targets the stopping-short and close-pass problems.
4. **Data augmentation / noise injection (PREP).** Add noise to the lidar during training to test
   robustness (the brief's "adversities"); or the DART method (expert with injected noise).
5. **PCA (W2-2).** Look at what the 36 lidar inputs span; probably low value for the report.
6. **Gaussian processes / uncertainty (W2-1, W3-1).** A model that knows when it is unsure could hand control
   back to a safe behaviour. Probably too much for the remaining time.
