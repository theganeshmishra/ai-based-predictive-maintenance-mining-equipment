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
| Data | CORA Vx vibration (3 kHz, 1 s windows, shaft-order features) | CORA infrared, every 3rd of 450 frames |
| Classes | healthy, bearing defect, half-broken rotor bar, broken rotor bar (4 runs each) | same |
| New data | 16 files (HB/OB vibration and infrared, about 1.06 GB), downloaded in a browser, all MD5-checked | |
| Shared data | healthy and bearing-defect runs are the same files as the Excavator's | |
| Images | — | 2,400 named images (`wheel_loader_<class>_<speed>_fNNN.png`), 600 per class |
| Main test | one whole speed held out per round (4 runs, one per class), 4 rounds, 16 run-level predictions | same |
| Code | `python Models/wheel_loader.py csv` | `python Models/wheel_loader.py cnn-pretrained` |

Following the methodology note on batch-transductive preprocessing, the Wheel Loader uses **no per-speed
normalisation**: preprocessing never looks at the held-out speed's runs.

## Results (accuracy, 4 classes; guessing = 0.250)

| Model | Main test (unseen speed) | Time split |
|---|---:|---:|
| Sensor: validation-selected model | 0.312 | — |
| Sensor: random forest | 0.250 | 0.938 |
| Sensor: XGBoost / LightGBM | 0.250 / 0.250 | 0.875 / 0.938 |
| Camera: pretrained ResNet18 | 0.312 | 0.812 |
| Anomaly: Isolation Forest (ROC-AUC) | 0.40 | 0.90 |

## What it means

- **At an unseen speed both branches are at chance** (5 of 16 runs). This is the same CORA limit as the
  Excavator: one run per fault per speed, so the model cannot learn what a rotor-bar fault looks like at a
  speed it has never heard.
- **Within the same recordings (time split)** the sensor model recognises 15 of 16 runs (all rotor-bar
  runs correct) and the camera 13 of 16, so the faults leave a signal; part of this score may come from
  recognising the recording.
- Broken rotor bars are classically detected in **motor current** (sidebands around the supply frequency),
  which this experiment does not use; CORA's current files (about 150 MB per run) are a possible next step.
- Small test set: one run changes accuracy by 6 points. Not above the methodology's 80% gate.
