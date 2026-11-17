# Attempt G results: gain-independent Grader sensor features

AI-Based Predictive Maintenance for Mining Equipment · Group 9 · November 16, 2026

Planned in [grader_gain_invariant_plan.md](grader_gain_invariant_plan.md), committed before the run (77d9bca), and run once. The completed experiment used the [archived verification runner](https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment/blob/48d95c50ad615eb4caa7f779d471ec981ea23058/Notebook/phase1_final_verification.py) with the `grader_gain` stage. Restore that recorded revision to reproduce it. Result file: [grader_gain_invariant.json](../Results/experiments/grader_gain_invariant.json).

## Result

Same fair test as the archived Grader sensor result: 120 JUST recordings, one working condition held out at a time, model chosen inside the training conditions. The only change is the feature set: 66 gain-independent features (crest factor, skewness, kurtosis and 8 relative band energies per channel), without RMS, standard deviation and peak. Logistic regression was chosen in all six rounds.

| | Archived model (all 84 features) | Attempt G (66 gain-independent features) |
|---|---:|---:|
| Accuracy (95% interval) | 84.2% (76.6–89.6) | **86.7% (79.4–91.6)** |
| Balanced accuracy / macro F1 | 84.2% / 84.7% | 86.7% / 86.6% |
| Recall: healthy, inner ring, outer ring, rolling element | 25/30, 25/30, 26/30, 25/30 | 30/30, 25/30, 27/30, 22/30 |
| Brier / ECE | 0.17 / 0.13 | 0.18 / 0.10 |

Disturbances on the held-out recordings (accuracy; same disturbances and seed as the final verification):

| Disturbance | Archived | Attempt G | Rule |
|---|---:|---:|---|
| Half amplitude | 29.2% | **86.7%** | at least 80%: pass |
| Double amplitude | 36.7% | **86.7%** | at least 80%: pass |
| Gaussian noise, 10% of RMS | 84.2% | **75.0%** | within 5 points: **fail** (−9.2) |
| 5% of samples removed | 89.2% | 85.8% | within 5 points: pass |
| First half only | 86.7% | 88.3% | pass |
| 90% rate read as full rate | 75.0% | 79.2% | pass |

## Decision (rule fixed in the plan)

Rule 1 (accuracy at least 82.5%) and rule 2 (half and double amplitude at least 80%) pass. **Rule 3 fails**: under added noise, accuracy falls 9.2 points below the archived model, more than the 5 points allowed. So nothing is deployed. `grader_sensor.joblib` and the Grader sensor route are unchanged, and the archived 84.2% stays the official result.

## What it shows

- The archived model's gain sensitivity comes from the three absolute-level features. Without them, a recording at half or double the level gives the same answer as the original (100% of recordings unchanged).
- The level features also carried useful information under noise. Without them, the shape statistics (kurtosis, crest factor) and the high-frequency bands are diluted by added noise, and accuracy drops to 75.0%.
- So the choice is a trade-off: the archived model needs a calibrated sensor with the same gain; attempt G needs a clean signal. Neither is robust to both. This was not resolved by tuning, because changing the features or the rule after seeing this result would make the test meaningless.
- A field system would need either a calibrated, fixed-gain accelerometer (archived model) or a noise level close to the laboratory recordings (attempt G). Choosing between them needs field data on which disturbance actually occurs.

## Input requirements this leaves in place

Neither the Grader nor the Drill Rig route can tell from the file whether the sampling rate or the units are right: the JUST and MCC5-THU CSVs carry no rate the routes use, and a wrong rate still looks like a plausible signal. Uploads must use the publisher's sampling rate (JUST 50 kHz, MCC5-THU 12.8 kHz) and units (m/s² for JUST). Non-finite values are already rejected by every sensor route.
