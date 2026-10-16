# Week 6: Dozer (machine 5) — track final-drive gear wear

**October 2026 — Saskatchewan Polytechnic PROJ 613.**

## Data: reused, not new

There is no public dozer dataset with matching sensor and camera data. As planned in the
[dataset manifest](../Dataset_Manifest/manifest.md), the Dozer final drive uses the **same 16 CORA
gear-wear runs as the Belt Conveyor** (CORA v2.1, DOI 10.34810/DATA2500, CC BY 4.0), with the Dozer's
planned target: **healthy vs gear-worn** (W25, W50 and W75 together). This is a different task on the
same rig data, not new evidence, and it is labelled that way everywhere.

| | Sensor | Camera |
|---|---|---|
| Data | CORA Vx vibration (3 kHz, 1 s windows) + T1 temperature | CORA infrared, every 3rd of 450 frames |
| Classes | healthy (4 runs), worn (12 runs) | same |
| Images | — | 2,400 named images (`dozer_<class>_<wear level>_<speed>_fNNN.png`): 600 healthy, 1,800 worn |
| Main test | one whole speed held out per round (1 healthy + 3 worn runs), 4 rounds, 16 run-level predictions | same |
| Code | `python Models/dozer.py csv` | `python Models/dozer.py cnn-pretrained` |

Because 12 of 16 runs are worn, **always guessing "worn" already scores 0.750**. Balanced accuracy
(the mean of the healthy and worn hit rates; guessing = 0.500) is the fair measure here.

## Results

| Model | Main: accuracy | Main: balanced accuracy | Time split: accuracy |
|---|---:|---:|---:|
| Guess "worn" (majority control) | 0.750 | 0.500 | 0.750 |
| Sensor: random forest | 0.812 | 0.625 | 1.000 |
| Sensor: validation-selected model | 0.625 | 0.583 | — |
| Sensor: XGBoost | 0.812 | 0.625 | 1.000 |
| Sensor: extra trees | 0.750 | **0.750** | 1.000 |
| Sensor: LightGBM | 0.750 | 0.583 | 1.000 |
| Camera: pretrained ResNet18 | 0.750 | 0.500 | 1.000 |
| Anomaly: Isolation Forest (ROC-AUC) | 0.63 | — | 0.92 |

## What it means

- At an unseen speed the sensor random forest finds all 12 worn runs but only 1 of the 4 healthy runs;
  the CNN calls every run "worn". Extra trees reaches the best balanced accuracy (0.750) but, with only 4
  healthy test runs and settings fixed in advance, this is one or two runs and not a reliable win.
- On the time split (first 60% of each run trains, last 30% tests) every model separates healthy from
  worn perfectly, the same pattern as the Belt Conveyor and Excavator.
- The anomaly results equal the Belt Conveyor's by construction (same healthy runs, same features).
- Small test set: 4 healthy runs, so one run changes the healthy hit rate by 25 points.
