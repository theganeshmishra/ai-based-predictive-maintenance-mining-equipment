# Week 6: Dozer (machine 5) — track final-drive gear wear

**October 2026 — Saskatchewan Polytechnic PROJ 613.**

## Data: reused, not new

There is no public dozer dataset with matching sensor and camera data. As planned in the
[dataset manifest](../Dataset_Manifest/manifest.md), the Dozer final drive uses the **same 16 CORA
gear-wear runs as the Belt Conveyor** (CORA v2.1, DOI 10.34810/DATA2500, CC BY 4.0). Following the
professor's remark ("4 class classification") it uses the **same 4 classes**: healthy, wear 25%, wear 50%
and wear 75%. The saved Dozer and Belt Conveyor scores matched because they
used the same inputs and setup; later image reruns can differ. This maps that
laboratory rig to the dozer final drive; it is **not separate evidence**.

(A first version used the manifest's 2-class target, healthy vs worn; it was replaced by the 4-class version.)

| | Sensor | Camera |
|---|---|---|
| Data | CORA Vx vibration (3 kHz, 1 s windows) + T1 temperature | CORA infrared, every 3rd of 450 frames |
| Classes | healthy, wear25, wear50, wear75 (4 runs each) | same |
| Images | — | 2,400 named images (`dozer_<class>_<speed>_fNNN.png`), 600 per class |
| Main test | one whole speed held out per round (4 runs, one per class), 4 rounds, 16 run-level predictions | same |
| Code | `python Models/dozer.py csv` | `python Models/dozer.py cnn-pretrained` |

## Results (accuracy, 4 classes)

| Model | Main test (unseen speed) | Time split |
|---|---:|---:|
| Majority class (control) | 0.250 | 0.250 |
| Sensor: validation-selected model | 0.625 | — |
| Sensor: random forest | 0.375 | 1.000 |
| Sensor: XGBoost | 0.438 | 1.000 |
| Sensor: LightGBM | 0.562 | 1.000 |
| Camera: pretrained ResNet18 | 0.562 | 1.000 |

All saved sensor and camera numbers are identical to the Belt Conveyor's, as expected.
These are the saved Dozer results. A later local rerun of the **Belt Conveyor**
image time split returned 0.8125 rather than its saved 1.000; the Dozer image
time split was not independently rerun. Neither time-split value demonstrates
performance on a new speed or a different dozer.
The anomaly-detection results belong to Week 7 and are not part of this Week 6 checkpoint.
