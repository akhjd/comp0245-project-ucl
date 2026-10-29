# Requirements traced to the 2026/27 brief

Source: COMP0245-Assignment-2026-2.pdf, supplied course brief, read locally. The PDF is not redistributed here.

- Goal (slides 2–4): behaviour cloning of an expert moving the robot end-effector to a Cartesian target; train a supervised regressor from robot state and target to wheel/arm commands. Generate expert trajectories, design evaluation and justify choices.
- Methodology (slide 5): consider data quantity/quality, optional additional inputs, instantaneous versus temporal inputs, architecture, training and hyperparameters. No specific architecture or dataset size is mandated in these slides.
- Evaluation (slide 6): consider accuracy, generalisation across targets/initial conditions, robustness to adversities and efficiency (memory, parameter count, computation).
- Timeline (slide 7): preliminary/final laptop demonstration in lab between 30 October and 6 November, unmarked but required for assessment completion; report and code deadline 6 November 2026. Confirm submission time and any later updates on Moodle.
- Report (slide 8): maximum six pages using the LaTeX IEEE template; introduction, methodology, results and conclusion, including justifications, comparisons, interpretation and limitations.
- Contribution statement (slide 9): maximum one page; may be attached as page seven; describe each group member's contribution.
- Code (slide 10): README explaining how to run and generated outputs, external code sources, and modified/original parts. Data and pretrained weights/checkpoints need not be included in the Moodle submission. A GitHub link in the report is allowed if code exceeds Moodle's file-size limit.

Repository organisation, numerical thresholds, obstacle scenarios, PyTorch, DAgger and split policies are implementation choices. They must be justified using actual experiments; they are not additional assignment requirements.
