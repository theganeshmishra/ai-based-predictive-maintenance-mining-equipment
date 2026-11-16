# Planned attempt G: gain-independent Grader sensor features (written before running)

AI-Based Predictive Maintenance for Mining Equipment · Group 9 · November 16, 2026 · committed before any result of this attempt existed

## Why

The final Phase 1 verification found that the Motor Grader sensor model (JUST slewing bearing) depends on the absolute vibration level: halving or doubling the held-out signals cut its fair-test accuracy from 84.2% to 29.2% and 36.7% ([report](phase_1_enhancement_report.md), section 8). A different accelerometer, gain setting or mounting would change that level, so the deployed model would fail on a correctly labelled but differently scaled recording. Three of its 14 features per channel are absolute levels in m/s²: RMS, standard deviation and peak. The other 11 do not change when the signal is multiplied by a constant: crest factor, skewness, kurtosis and the 8 relative log band energies.

## G. Grader sensor, gain-independent feature set

- **Data and test:** unchanged from the archived result: 120 JUST recordings, 1 s windows, hold out one working condition at a time (6 rounds of 20 recordings), the same five candidate models chosen inside the training conditions, one prediction per recording, majority control.
- **Only change:** the feature set offered is the 66 gain-independent features (crest, skewness, kurtosis and band0 to band7 for each of the 6 channels). RMS, standard deviation and peak are not offered. The feature extractor itself is unchanged.
- **Robustness:** the same disturbances as the final verification (seed 613, refit per round on the training conditions, only held-out recordings disturbed): Gaussian noise at 10% of RMS, half and double amplitude, 5% of samples removed, first half only, 90% rate read as full rate.
- **Runs once.** No change to the feature list, models or rule after seeing the result.

## Decision rule (fixed now)

The gain-independent model replaces the Grader sensor route only if **all** of these hold:

1. Fair-test accuracy is at least 82.5% (99 of 120, at most two recordings below the archived 84.2%).
2. Accuracy under half amplitude and under double amplitude are each at least 80%.
3. No other disturbance drops accuracy more than 5 points below the archived model's result for that disturbance (noise 84.2, 5% removed 89.2, first half 86.7, 90% rate 75.0).

If all hold, the same selection is fitted on all 120 recordings and saved as a **new file**, `grader_sensor_gain_invariant.joblib`, registered in the local trust list. `predict` uses it for JUST CSVs. The existing `grader_sensor.joblib` stays on disk, unchanged, for comparison and rollback. The archived result in `Results/grader/sensor.json` is not changed; the new result is reported next to it.

If any condition fails, the result is reported and nothing is deployed.

Results go to `Results/experiments/grader_gain_invariant.json`.
