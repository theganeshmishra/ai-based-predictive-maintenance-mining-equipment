# Week 7: all available sensor and image data

**October 26–27, 2026 — Saskatchewan Polytechnic PROJ 613.** Goal: use every sensor and image source the
public datasets offer for each machine, not only the subsets used so far. The Week 6 results are unchanged;
every new run writes separate result files (`*_fulldata.json`, `*_allframes.json`).

## 1. New data

- **CORA (Belt Conveyor, Dozer, Excavator, Wheel Loader):** all 1,806 remaining CORA files were downloaded
  (about 37 GB; every file checked against the publisher's size and MD5; local only in
  `Raw_Data/cora_full/`, not in Git). Now available for all 9 conditions × 4 speeds:
  - stationary runs: vibration X/Y/Z, motor current C1–C3, six temperature probes T1–T6, speed, room temperature;
  - start-up tests: **five 30 s repeats** per condition and speed with vibration X/Y/Z, current C1–C3 and speed.
- **Images:** every infrared frame (450 per CORA run instead of every 3rd) and the **whole** event-camera
  recording for the Haul Truck (16–26 s instead of the first 5 s).
- **Pump and Haul Truck sensors** already used every recording and window of their datasets (Ottawa 48 × 10 s;
  UODS 30 × 10 s with accelerometer and microphone), so nothing more was available there. The Pump camera was
  not extended: it is already 100% on the main test, and its 48 GB of video would add hours of CPU with no
  room to improve.

## 2. CORA sensors: every sensor, window and recording

New command `python Models/<machine>.py fulldata` (engine: `Models/cora_fulldata.py`). Every 1 s window of
every channel; features per window only (vibration in shaft orders, current in supply-frequency orders, so no
preprocessing looks at the held-out speed). **Main test:** each whole speed held out in turn; training uses the
stationary runs *and* start-up tests of the other three speeds; the held-out speed gives **24 test recordings
per round (96 in total, was 16)**. Feature set and model (random forest or LightGBM) chosen by
leave-one-speed-out validation inside the training speeds.

| Machine | Week 6 (16 runs) | Full data, validation-selected (96 recordings) | stationary runs only | start-up tests only | Time split (stationary) |
|---|---:|---:|---:|---:|---:|
| Belt Conveyor | 0.625 | 0.521 | 0.625 | 0.500 | 1.000 |
| Excavator | 0.188 | 0.375 | 0.438 | 0.362 | 1.000 |
| Wheel Loader | 0.312 | 0.406 | 0.500 | 0.388 | 1.000 |
| Dozer* | 0.625 | 0.521 | 0.625 | 0.500 | 1.000 |

Guessing = 0.250. \*Same CORA runs and classes as the Belt Conveyor, so identical by construction.

- More data helps the Excavator (0.188 → 0.375) and Wheel Loader (0.312 → 0.406); **motor current** is the
  feature set most often chosen (it carries the classic rotor-bar and bearing signatures). The Belt Conveyor
  stays about the same (its Week 6 0.625 used per-speed normalisation, which looked at the held-out runs).
- The larger test set (96 recordings) makes these numbers more reliable: one recording now moves accuracy by
  about 1 point instead of 6.
- **No CORA machine reaches 80–90% on unseen speeds.** Within the same recordings (time split) every machine is
  at 0.94–1.00.

### Temperature: a recording-session shortcut, not a fault signal

With the six temperature probes added, unseen-speed accuracy jumped to 0.75–1.00. A check shows why: the
**pattern between the probes already identifies the condition in the first 5 minutes of a run**, before any
fault-related heating can build up, while the actual warming since the start of the run does not.

| Temperature-only check (16 runs, unseen speed) | Belt Conveyor | Excavator | Wheel Loader |
|---|---:|---:|---:|
| probe pattern, first 5 minutes only | 0.812 | 0.938 | 0.750 |
| probe pattern, whole run | 0.875 | 0.875 | 0.750 |
| warming since run start (fault heating) | 0.375 | 0.375 | 0.375 |

CORA rebuilt the rig once per condition and then ran all four speeds, so the probe pattern most likely reflects
that **assembly session** (probe placement, residual heat), not the fault. Temperature sets are therefore still
reported in the result files but are **excluded from model selection**. The same session issue can affect every
CORA modality to a smaller degree: each condition was assembled once, so the unseen-speed test cannot fully
separate "fault" from "assembly session". Repeated assemblies per condition would be needed to settle it.

### XGBoost / LightGBM and anomaly detection on the full data

Both comparisons were rerun on the full data, using the same fair main test: each whole speed is held out in
turn, giving 96 test recordings. The inputs are all vibration and current features (temperature is excluded).
Every model setting was fixed in advance, so nothing was tuned on a test result. Command:
`python Models/<machine>.py fulldata --part ensembles` (or `--part anomaly`, or `--part all`).

| Main test accuracy (unseen speed) | Belt Conveyor / Dozer | Excavator | Wheel Loader |
|---|---:|---:|---:|
| Majority class (control) | 0.250 | 0.250 | 0.250 |
| Random forest | 0.573 (Week 6: 0.375) | 0.438 (0.250) | 0.333 (0.250) |
| Extra trees | 0.500 (0.438) | **0.458** (0.312) | 0.323 (0.250) |
| Gradient boosting (scikit-learn) | **0.594** (0.438) | 0.417 (0.250) | 0.323 (0.250) |
| XGBoost | 0.531 (0.438) | **0.458** (0.250) | 0.292 (0.250) |
| LightGBM | 0.583 (0.562) | 0.427 (0.312) | **0.365** (0.250) |
| Time split (secondary), all five models | 1.000 | 1.000 | 0.938–1.000 |

- With more data, every real model improves on its Week 6 score for every machine.
- No model wins on all three machines, and the gaps between models (about 2–9 points) are only a few
  recordings. The best-model figures in bold are therefore **not** a fair pick. The fair single numbers are
  the validation-selected results in the first table of this section.
- Still no CORA machine reaches 80–90% on an unseen speed.

| Anomaly detection, ROC-AUC (healthy vs fault) | Belt Conveyor / Dozer | Excavator | Wheel Loader |
|---|---:|---:|---:|
| Isolation Forest, main test (Week 7 small data) | 0.63 (0.63) | 0.56 (0.42) | 0.53 (0.40) |
| One-Class SVM, main test | 0.66 (0.58) | 0.65 (0.50) | 0.59 (0.50) |
| Time split (secondary), IF / OC-SVM | 0.94 / 0.83 | 0.83 / 0.73 | 0.75 / 0.60 |

- The detectors learn healthy data only. The alarm threshold is set inside the training speeds: fit on two
  speeds, then use the highest healthy score on the third. No test data or fault labels are used.
- **The alarm does not work on an unseen speed.** The One-Class SVM flags all 24 recordings at the 5 Hz speed
  (healthy or not) and none at the other three speeds, so it detects 25% of faults with 25% false alarms. The
  Isolation Forest never alarms. The detector reacts to the change in speed, not to the fault. ROC-AUC is
  only slightly above chance (0.53–0.66).
- A usable alarm would need healthy reference data at the operating speed. That is the normal practice in
  real monitoring (a baseline per operating point), but it falls outside the fair unseen-speed test.

## 3. Images: every frame / whole recordings

| Machine | Week 6 images | Main test (unseen speed) | Time split |
|---|---|---:|---:|
| Belt Conveyor | 2,400 (every 3rd frame) → 7,200 (all) | 0.562 → **0.500** | 0.812 → 0.938 |
| Excavator | 2,400 → 7,200 | 0.312 → **0.250** | 0.938 → 1.000 |
| Wheel Loader (Ganesh, Week 7) | 2,400 → 7,200 | 0.312 → **0.188** | 0.812 → 1.000 |
| Haul Truck | 463 (first 5 s) → 1,355 (whole recordings) | 0.417 → **0.750** | 0.913 → 1.000 |

- **CORA infrared:** more frames from the same runs do not help on an unseen speed; neighbouring frames are near
  copies, so they add images but not independent examples. Only the easier time split rises.
- **Haul Truck event camera: a real improvement.** Whole recordings give the CNN 3× more and more varied motion
  images; at the unseen 2000 rpm speed it now recognises 9 of 12 recordings (healthy 3/3, ball 3/3, outer 2/3,
  inner 1/3), up from 5 of 12. Same caveat as before: 12 test recordings, and each bearing was installed once and
  recorded at both speeds.

Commands: `python Models/<machine>.py export-images --all-frames`, then
`python Models/<machine>.py cnn-pretrained --all-frames` (add `--split time` for the time split).

## 4. Summary

| | Before | With all available data |
|---|---|---|
| Pump | sensor 0.958, camera 1.000 | unchanged (already all data / already 100%) |
| Haul Truck | sensor 0.800–0.867, camera 0.417 | sensor unchanged (all data already used), **camera 0.750** |
| Belt Conveyor / Dozer | sensor 0.625, camera 0.562 | sensor 0.521 on 96 recordings, camera 0.500 |
| Excavator | sensor 0.188, camera 0.312 | **sensor 0.375**, camera 0.250 |
| Wheel Loader | sensor 0.312, camera 0.312 | **sensor 0.406**, camera 0.188 |

All results: `Results/<machine>/<machine>_fulldata.json` and `*_allframes*.json`; figures in `Figures/<machine>/`.
