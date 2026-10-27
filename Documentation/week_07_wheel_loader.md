# Week 7: Wheel Loader (machine 6) — electric drive motor

**October 2026 — Saskatchewan Polytechnic PROJ 613.**

## Why this data

A search for a separate public dataset with sensor and camera recordings of the same gearbox,
transmission or bearing faults found none (vibration-only sets such as SpectraQuest/OEDI, XJTU gearbox,
KAIST and Ottawa motor data have no camera images; the *Scientific Data* 2026 paper on a multimodal
electromechanical bench describes CORA itself). Instead of reusing the Belt Conveyor's gear-wear runs a
third time, the Wheel Loader uses **CORA faults no other machine in this project uses**: half-broken and
fully broken **rotor bars** of the induction motor. Large wheel loaders can use electric drive, so the motor
stands in for the loader's drive motor. Lab data, not a wheel loader.

| | Sensor | Camera |
|---|---|---|
| Data | CORA Vx vibration (3 kHz, 1 s windows, shaft-order features) | CORA infrared, all 450 frames per run |
| Classes | healthy, bearing defect, half-broken rotor bar, broken rotor bar (4 runs each) | same |
| New image data | 8 HB/OB infrared stacks, downloaded and MD5-verified | |
| Shared data | healthy and bearing-defect runs are the same files as the Excavator's | |
| Images | — | 7,200 named images (`wheel_loader_<class>_<speed>_fNNN.png`), 1,800 per class |
| Main test | one whole speed held out per round (4 runs, one per class), 4 rounds, 16 run-level predictions | same |
| Code | `python Models/wheel_loader.py csv` | `python Models/wheel_loader.py cnn-pretrained` |

Following the methodology note on batch-transductive preprocessing, the Wheel Loader uses **no per-speed
normalisation**: preprocessing never looks at the held-out speed's runs.

## Results (accuracy, 4 classes; guessing = 0.250)

All 16 infrared stacks were downloaded and verified against the publisher's file sizes and MD5 checksums. The exporter now saves all 450 frames per run and the ResNet18 was retrained on all 7,200 images. The older saved result below has been replaced.

The eight HB/OB vibration CSVs were missing on the laptop used for the image rerun, so the sensor scores were not retrained there. On October 26, 2026 they were re-run on a laptop with all 16 MD5-verified vibration files: the CSV, ensemble and anomaly scores (main test and time split) reproduced the saved values exactly.

| Model | Main test (unseen speed) | Time split |
|---|---:|---:|
| Sensor: validation-selected model | 0.312 | — |
| Sensor: random forest | 0.250 | 0.938 |
| Sensor: XGBoost / LightGBM | 0.250 / 0.250 | 0.875 / 0.938 |
| Camera: pretrained ResNet18 (7,200 images) | 0.188 | 1.000 |
| Anomaly: Isolation Forest (ROC-AUC) | 0.40 | 0.90 |

## What it means

- **At an unseen speed the camera gets 3 of 16 recordings (0.188), below the 0.250 four-class chance
  baseline**; recall is zero for broken-bar faults. The previous 2,400-image run scored 5 of 16, so
  adding adjacent frames did not improve generalization. There is still only one recording per class
  and speed; images from one recording are correlated, not independent examples.
- **The secondary time split gets 16 of 16 recordings (1.000) and 0.911 image-level accuracy**, but
  train and test frames come from the same recordings. This is not evidence of generalization to a new
  recording or speed and may reflect recording-specific patterns.
- Broken rotor bars are classically detected in **motor current** (sidebands around the supply frequency),
  which this experiment does not use; CORA's current files (about 150 MB per run) are a possible next step.
- Small test set: one run changes accuracy by 6 points. Not above the methodology's 80% gate.
