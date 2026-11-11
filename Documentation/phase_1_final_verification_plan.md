# Phase 1 final verification: planned checks and one planned experiment (written before running)

AI-Based Predictive Maintenance for Mining Equipment · Group 9 · November 11, 2026 · committed before any of these results existed

This file fixes, in advance, what the final Phase 1 verification runs. It is a verification pass: archived results and deployed checkpoints are not changed, and nothing here is promoted to deployment. Outputs go to `Results/experiments/phase_1/final_verification.json` and `Figures/experiments/phase_1/`. The runner is `Notebook/phase1_final_verification.py`.

## 1. Reproduction and calibration of the archived sensor fair tests (all eight mappings)

- Rebuild every sensor feature table from the raw files and rerun the archived fair test exactly (same folds, same candidate models and feature sets, model chosen inside the training groups).
- **Reproduced** means the recording-level confusion matrix and the chosen (feature set, model) per round are identical to the archived `Results/<machine>/sensor.json`.
- Save each test recording's mean class probability. Report, per mapping: multiclass Brier score, top-label expected calibration error (5 equal-width bins), mean top-label confidence for correct and wrong predictions, and a Wilson 95% interval for accuracy. With 16 to 120 test recordings these are descriptive only; no calibration is fitted and deployed outputs are not changed.

## 2. Leakage audit

- Sensor tables: each recording has one class and one held-out group; each round holds out exactly its declared groups; no recording is on both sides; every recording is tested once across rounds; identical feature rows shared between different recordings are counted.
- Images: for every exported image index, exact duplicate files (SHA-256) are checked across each round's train and test sides, and whole recordings or source units must not cross the split. Near-duplicates that are not byte-identical are not detected by this check.

## 3. Sensor robustness beyond the Pump

For Haul Truck (UODS), Drill Rig (MCC5-THU) and Motor Grader (JUST): in each archived fair-test round, the archived chosen (feature set, model) is refitted on the training groups, and only the held-out recordings are disturbed. The disturbances are the Pump set, unchanged: Gaussian noise at 10% of each channel's RMS (seed 613), half and double amplitude, random removal of 5% of samples, the first half of the recording, and resampling to 90% of the rate while the extractor still assumes the original rate. Multi-channel recordings get the same removal mask on every channel. Report accuracy, balanced accuracy, macro F1 and the share of recordings whose class is unchanged from the clean prediction. The clean case must reproduce the archived score. These are synthetic disturbances on laboratory data, not a field robustness claim.

## 4. Planned experiment E: CORA transient runs, unseen speed

- **Why:** the CORA stationary fair test has one run per class and speed, so it cannot tell whether the failure at an unseen speed comes from too few runs or from the speed change itself. The transient start-up runs (5 repetitions per class and speed, already downloaded for attempt C) have never been used for an unseen-speed test.
- **Test:** Belt Conveyor, Excavator and Wheel Loader classes (Dozer reuses the Belt Conveyor runs). Four rounds, each holding out one speed (F5, F15, F50, F60) with all five repetitions, so 20 test runs per round and 80 in total. Attempt C's transient features (orders normalised with the measured speed), the same five candidate models, chosen inside the training speeds. One prediction per run, with the majority control.
- **Runs once.** No change after seeing the result.
- **Reading:** clearly above the 25% control would mean more runs per class help at an unseen speed; near the control would mean the speed change itself is the barrier. Either way the result is reported only. It is not deployed, because the deployed routes take stationary recordings, and it does not replace the archived stationary fair test.

## 5. Other checks

- Rerun the committed Phase 1 runner's CORA candidates, Pump robustness and deployment probes on this computer and compare with `Results/experiments/phase_1/verification.json`.
- Recompute the reported JUST diagnostic values from the original archive member; record sampling rates that can be measured from time columns.
- CPU inference benchmark of the Pump sensor, Pump image and Aalto routes on this Windows computer, one fresh process per route, with peak memory.
- New automated tests: saved confusion matrices agree with every reported metric; scalers sit inside the model pipelines; model selection never sees test rows; the image model cache keeps routes apart and stays bounded; analytic checks of skewness, kurtosis, crest factor, peak-to-peak, Welch band power and scale stability.
