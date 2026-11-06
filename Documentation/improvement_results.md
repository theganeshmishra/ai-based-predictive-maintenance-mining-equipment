# Improvement attempts: results

PROJ 613 · Group 9 · November 6, 2026

The team asked whether more machines can reach 80% accuracy. The three attempts were fixed in advance in the [improvement plan](improvement_plan_2026-11-06.md) (committed before any were run), and each was run once. All three results are reported below. The archived fair-test results and the deployment checkpoints are unchanged. The new results are in `Results/experiments/improvement/` and the figures in `Figures/experiments/improvement/`.

Rerun commands: `python Notebook/pdm.py attempt drill-orders`, `attempt haul-frames`, `attempt cora-transient`.

## Summary

| Attempt | Test | Before | After | Reaches 80%? | Decision |
|---|---|---|---|---|---|
| A. Drill Rig sensor, tachometer-order features | unseen speed (same fair test) | 73.3% acc, 68.1% bal | **77.5% acc, 75.7% bal**, macro F1 78.2% | no | better under the same protocol; recommended, checkpoint not replaced |
| B. Haul Truck images, speed-scaled frames | unseen speed (same fair test) | 58.3% archived; 66.7% rerun | 50.0% | no | no gain; archived 58.3% stays official |
| C. CORA transient runs | unseen repetition at a trained speed (narrower test) | n/a | Belt Conveyor 100%, Excavator 100%, Wheel Loader 97.5% | yes, under this narrower test only | reported separately; does not replace the fair test |

## A. Drill Rig sensor: tachometer-order features

- **Data:** 120 MCC5-THU recordings, 7,080 one-second windows. The `speed` column gives one pulse per shaft revolution. Each window's shaft rate is interpolated from the pulse times, and each vibration channel's spectrum is summed in shaft-order bands (0.5–1.5, 1.5–2.5, 2.5–4.5, 4.5–8.5, 8.5–16.5, 16.5–32.5, 32.5–64.5, 64.5–120.5). The shaft rate itself is not a feature.
- **Test:** the archived fair test, holding out 1000, 2000 and 3000 rpm in turn. The feature set (`vibration`, `orders` or `vibration+orders`) and the model are chosen inside the training speeds.
- **Result:** 93 of 120 recordings correct. Accuracy 77.5%, balanced accuracy 75.7%, macro F1 78.2%. The majority control scores 30.0%.
- **Confusion matrix** (rows actual, columns predicted; healthy, gear wear, gear pitting, tooth crack): `[[8,4,0,0],[0,36,0,0],[0,9,24,3],[0,10,1,25]]`. Healthy recall rose from 5 of 12 to 8 of 12. Most remaining errors are pitting or crack called gear wear.
- **Chosen per round:** vibration+orders with gradient boosting, vibration+orders with logistic regression, and vibration with gradient boosting. So the order features were picked in two of the three rounds.
- **Decision:** a real improvement under the same protocol (+4.2 points accuracy, +7.6 points balanced), but still under 80%, and on only 120 recordings. Under the team's no-retrain rule, the deployment checkpoint is not replaced. We recommend order features for any future retraining.

## B. Haul Truck images: frames covering the same number of turns

- **Change:** the 2000 rpm test recordings were cut into 62.5 ms frames instead of 125 ms, so each frame covers about 2.1 revolutions at both speeds. Training frames (1000 rpm, 125 ms) are unchanged.
- **Baseline rerun with the current code (125 ms):** 8 of 12 test recordings correct, 66.7% accuracy, macro F1 67.6%.
- **Speed-scaled frames:** 6 of 12 correct, 50.0% accuracy, macro F1 47.5%. Healthy stays 3 of 3. More faulty recordings are now called healthy.
- **Archived result:** 58.3% (7 of 12). The rerun differs from the archive by one recording. The difference is run-to-run variation, not an improvement, so the archived number stays official.
- **Finding found while checking this run:** the epoch choice uses one training recording per class as validation, and each of those is a camera angle that its class never shows in the remaining training data. At every epoch the model gets all 8 remaining training recordings right and all 4 validation recordings wrong (image accuracy 0.0–0.4%). So the epoch choice always falls to 1 epoch, and the model is largely recognising the camera angle (scene), not the fault. This applies to the archived Haul Truck image result too, and it is the main reason we do not expect frame-length changes to help. It is a limitation of the event-camera source (three angles per class), not a code error.
- **Decision:** no gain. No change to the model or the archive.

## C. CORA transient runs: unseen repetition at a known speed

- **Data:** CORA transient start-up runs from CSUC Dataverse doi 10.34810/DATA2500 (CC BY 4.0). There are 5 repetitions per class and speed, each 30 s of Vx at 3 kHz with the output-shaft RPM at about 100 Hz. Each 1 s window is normalised by its measured speed, and windows below half the steady speed are dropped. The 396 files are listed with SHA-256 in [`cora_transient_files.json`](../Dataset_Manifest/cora_transient_files.json).
- **Test:** five rounds, each holding out repetition k at all four speeds (16 runs per round, 80 in total). The model is chosen inside the training repetitions. One prediction per run.

| Machine (classes) | Runs | Accuracy | Balanced | Macro F1 | Majority control | Errors |
|---|---|---|---|---|---|---|
| Belt Conveyor (healthy, W25, W50, W75) | 80 | 100% | 100% | 100% | 25% | none |
| Excavator (healthy, bearing, unbalance, misalignment) | 80 | 100% | 100% | 100% | 25% | none |
| Wheel Loader (healthy, bearing, half-broken bar, broken bar) | 80 | 97.5% | 97.5% | 97.5% | 25% | 2 half-broken-bar runs called broken bar |

- **What it shows:** the CORA features separate the classes well when the model has seen the class at that speed. The fair-test failure (18.75–56.25%) comes from the change of speed, not from features that carry no fault information.
- **What it does not show:** this is not the unseen-speed claim. The repetitions use the same physical gear set or fault and the same session, so session effects cannot be ruled out. It is not a new-specimen test either. The Dozer reuses the Belt Conveyor runs, so its result is the Belt Conveyor's.

## Which machines reach 80%, and under which test

The official headline for each machine stays its fair test (whole speeds, bearings, designs or supports held out). The other tests answer narrower questions and must be labelled as such.

| Machine | Fair test, sensor | Fair test, images | ≥ 80% on the fair test? | Narrower tests that reach 80% |
|---|---|---|---|---|
| Pump | 95.8% | 100% | yes (both) | |
| Grader | 84.2% | 97.2% acc, 82.6% bal (per tooth crop) | yes (both; separate tasks) | |
| Drill Rig | 73.3% (attempt A: 77.5%) | 88.9% (CAD renders) | images only | |
| Haul Truck | 80.0% acc, 73.3% bal | 58.3% | sensor accuracy only, just at the line; balanced is below | without the load-confounded ball class: 84.0% acc, 73.3% bal |
| Belt Conveyor | 43.8% | 50.0% | no | time split 100% / 100%; transient repetition 100% |
| Dozer (= Belt Conveyor runs) | 43.8% | 56.3% | no | time split 100% / 93.8%; transient = Belt Conveyor |
| Excavator | 25.0% | 25.0% | no | time split 93.8% / 100%; transient repetition 100% |
| Wheel Loader | 31.3% | 18.8% | no | time split 93.8% / 100%; transient repetition 97.5% |

**Bottom line:** under the fair test, four of eight machines reach 80% on at least one modality (Pump, Grader, Drill Rig images, and Haul Truck sensor accuracy at exactly 80%). The four CORA machines do not, and none of the planned attempts changes that. CORA reaches 97.5–100% only when the test speed was seen in training. We report that as a separate, narrower result, not as the headline.
