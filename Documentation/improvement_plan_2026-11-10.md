# Planned improvement attempt D (written before running)

PROJ 613 · Group 9 · November 10, 2026 · committed before any result of this attempt existed

The team wants at least 80% accuracy. Under the fair test, the Drill Rig sensor model is the closest miss: 73.3% archived, 77.5% with attempt A's shaft-order features. 80% means 96 of the 120 recordings correct, 3 more than attempt A. This file fixes one more attempt in advance. It runs once and is reported whatever the result. The archived results and the deployed checkpoint stay unchanged; new results go to `Results/experiments/improvement/`.

## Why this attempt

Attempt A's confusion matrix (an existing result, used here only to choose what to try): 19 of the 72 pitting and tooth-crack recordings were called gear wear. Pitting and cracks are local faults that hit once per turn of the damaged gear; wear is spread around the gear. The standard way to see such repeated impacts is the envelope spectrum: high-pass the vibration, take its envelope, and look at which shaft orders the envelope repeats at.

## D. Drill Rig sensor: envelope order features

- **Data:** the same 120 single-fault MCC5-THU recordings and the same 1 s windows (12,800 samples) as attempt A.
- **New features, per vibration channel and window (9 per channel, 54 in all):**
  - The whole recording is high-pass filtered (4th-order Butterworth, 1 kHz cutoff, zero-phase), then its Hilbert envelope is taken. The cutoff is fixed here and is not tuned.
  - Per window, the envelope's mean is removed, a Hann window applied, and its power summed in the same shaft-order bands as attempt A (0.5–1.5 … 64.5–120.5), using attempt A's shaft rate from the `speed` tachometer. Each band is a log share of the total, like attempt A. The shaft rate itself is not a feature.
  - The kurtosis of the high-passed signal in the window.
- **Feature sets offered (fixed now):** `vibration` (archived), `vibration+orders` (attempt A) and `vibration+orders+envelope`.
- **Test:** the same fair test as before: hold out 1000, 2000 and 3000 rpm in turn. The same five candidate models. The feature set and model are chosen inside the training speeds by macro F1 on validation groups. One prediction per recording (mean window probability). Accuracy, balanced accuracy and macro F1, with the majority control, compared with the archived 73.3% and attempt A's 77.5%.
- **Runs once.** No change to the cutoff, bands, feature sets or models after seeing the result.

## Decision rule (fixed now)

- **If the fair test reaches at least 80% accuracy** and its balanced accuracy is above the archived 68.1%: the same procedure is fitted on all 120 recordings and saved as a separate file, `drill_rig_sensor_envelope.joblib`. `predict` uses it only for MCC5-THU CSVs that have the `speed` column, and says so in its answer. The fair-test number is the accuracy estimate for that procedure. If the all-data selection picks a feature set without the new features, the result is the existing model and nothing is deployed (as happened after attempt A). `drill_rig_sensor.joblib` is not changed in any case.
- **If it stays under 80%:** the result is reported next to attempts A and the archived number, nothing is deployed, and no further Drill Rig sensor attempts are made with this data.
