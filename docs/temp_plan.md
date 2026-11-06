# Temporary plan

Draft from 2 Nov 2026, and likely to change. It is based on review notes the user pasted; the notes were written from a summary, not the code. The references to Osa and to the slides and brief come from those notes and have not been checked here.

## Where we are

- **Done:** expert, dataset, `mlp1` and `mlp2` training, DAgger code.
- **Running at last check:** round 1 validation of `runs/dagger1`.
- **Best sign so far:** on the first 14 validation runs, median closest hand distance fell from 2.55 m to 0.29 m, with 0 dots touched.
- **Main gap:** no model reaches dots yet.

## Decisions needed from the user

| Decision | Options | Note |
|---|---|---|
| Velocity input | Keep, remove, add noise or dropout, or predict the change in wheel speed | Velocity is one of the user's chosen inputs. Lidar history may still leak ego-motion after removal |
| Dot input | Keep all remaining dots, give the current target, or sort slots in expert order | Decisions 18-20 chose all remaining dots in fixed order, so the others could only be ablations unless the user changes this |
| A* waypoint input | Leave out, or test as a clearly labelled ablation | The user ruled out A* inputs for the main MLP |
| Arm control | Keep wheels only, or also learn the arm | The notes say the brief's outputs include arm commands. Ask the lab staff whether wheels only is acceptable |
| Lidar and dot features | Try or skip | See `docs/MLP_DIAGNOSIS_AND_INPUT_FEATURES.md` |

## Experiments

### 1. Get a working closed-loop model
- Finish the DAgger rounds, or stop if round 1 is still near zero.
- Plot driving metrics against DAgger round. This is likely the main report figure.
- Run the `mlp2` driving test, so the augmentation has a result.
- Consider expert handover, or mixing expert and learner with a decaying beta. DAgger is currently pure MLP after the starting model, which may limit how far runs get.

### 2. Test the velocity shortcut by retraining
Hiding velocity in `mlp1` only shows what that model relies on, not what a retrained model could do.
- Train without velocity.
- Check whether lidar history still leaks it, for example with 1 vs 5 frames.
- If copying remains, try predicting the change in wheel speed.

### 3. Ablations
Use the same seeds and splits, and retrain each one:
- velocity in or out
- lidar history of 1 vs 5 frames
- MSE vs Huber or L1
- dataset size
- network width and depth
- observability: current target, expert-ordered slots, next A* waypoint

### 4. Baselines
- Linear or tree-based regressor.
- Non-learned controller, to show the MLP is warranted.

## Evaluation

| Question | What to report | Current state |
|---|---|---|
| Accuracy | Dots reached, contact time, time to dot, path length vs expert | Dots, contact, time and path are in each run's `summary.json`; add the expert comparison |
| Generalisation | Train on cases 1-4 and test on 5; more cylinders, faster movers, a new floor plan | Not done |
| Robustness | Lidar noise and dropout, actuator noise, perturbed start poses | `rollout` has lidar noise, dropout, delay and push options; not connected to the new MLP test |
| Efficiency | MLP inference time vs expert planning time and the 1 ms control period | Parameters known; `inference_ms` is saved per row but not summarised |
| Error analysis | Error near obstacles vs cruising; MLP query rate vs 100 Hz recording | Not done. MLP currently commands every 10 ms |

Also:
- Report per-case results. `evaluation.json` already has them for each round.
- Add seeds and confidence intervals, because 0/82 and 1/112 are small counts.
- Choose the final policy using validation. `comp0245/training/dagger_mlp.py` already picks the best round this way.
- Run the test split once, at the end, on the final model only.

## Submission

- Commit `comp0245/training/train_mlp.py`, `report_progress.txt`, the diagnosis doc and this plan if wanted.
- README: run commands, outputs, and which code was provided vs written or modified (RRTExpert, environment, DAgger).
- Report: 6 pages in the IEEE LaTeX template, plus the 1-page contribution statement.
- State these limits plainly in the report:
  - The MLP outputs wheels only, and the expert controls the arm.
  - The expert sees more than the MLP: full layout, A* routing, dot order and its internal plan.
  - The clean-run filter may under-represent the hardest scenes.
  - High offline R² did not mean good driving. This is a useful finding on its own.
- Lab demo, 30/10-06/11: not marked but required for completion, so show something this week. Confirm the exact session.

## Suggested order

1. Finish or stop DAgger, and decide the input fix.
2. Retrain the key ablations with the same seeds and splits.
3. Run closed-loop evaluation per case, then robustness and efficiency.
4. Run the test split once, then write the report and README.

## Sources

- Osa et al., 2018, *An Algorithmic Perspective on Imitation Learning*: https://arxiv.org/pdf/1811.06711
- Zare et al., 2023, *A Survey of Imitation Learning*: https://arxiv.org/pdf/2309.02473
