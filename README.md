# AI-Based Predictive Maintenance for Mining Equipment

Saskatchewan Polytechnic PROJ 613 capstone · Ganesh Mishra, Lakshya Sharma and Sagar Patel

For each machine we detect the **current fault condition of one component** two separate ways, from a **sensor** (vibration or temperature, as a CSV of features) and from a **camera** (CNN on images), and compare them. All data are public laboratory datasets used as stand-ins for mining equipment, so the results show the method, not a validated mine-site detector.

## Results (accuracy, 4 classes)

**Week 6 presentation scope:** the five machine mappings below. Dozer reuses Belt Conveyor data; Wheel Loader, Drill Rig and Grader remain planned. Existing Week 7 anomaly work is outside this checkpoint.

| Machine (component) | Classes | Sensor | Images (CNN) | Sensor, time split | Images, time split |
|---|---|---:|---:|---:|---:|
| **Pump** (support bearing) | healthy, inner, outer, ball | **0.958** | **1.000** | — | — |
| **Haul Truck** (wheel-motor bearing) | healthy, inner, outer, ball | 0.800–0.867 | 0.417 | 1.000 | 0.913 |
| **Belt Conveyor** (geared drive) | healthy, wear 25/50/75% | 0.625 | 0.562 | 1.000 | 0.812 rerun (1.000 saved) |
| **Excavator** (swing drive) | healthy, bearing, unbalance, misalignment | 0.188 | 0.312 | 0.938 | 0.938 |
| **Dozer** (track final drive)* | healthy, wear 25/50/75% | 0.625 | 0.562 | 1.000 | 1.000 |

- **Main test (columns 3–4):** whole recordings, bearings or speeds the model never saw in training. This is the primary grouped result, subject to the CORA preprocessing caveat below.
- **CORA preprocessing caveat:** the later four-class Belt Conveyor/Dozer and some Excavator sensor folds normalize features across all recordings at a speed, including the held-out speed. No test labels enter fitting, but these are batch-transductive scores, not estimates for one entirely isolated new machine. See [methodology](Documentation/methodology.md).
- **Time split (columns 5–6):** the model trains on the first 60% of each recording and is tested on the last 30%. This test is easier because the same recordings appear in training, so it is reported separately.
- **Week 6 reproduction note:** on October 23, the published Belt Conveyor time-split sensor command reproduced 1.000, but the published image command produced 0.8125 (13/16 runs) instead of its earlier saved 1.000. The image score is not exactly reproducible in this local rerun; see the [Week 6 summary](Documentation/week_06_professor_remarks.md). The whole-speed image result remains 0.5625. The Dozer time-split image number below is the earlier saved score on the same CORA runs, not an independent rerun.
- \*The Dozer reuses the **same 16 CORA runs, features and 4 classes as the Belt Conveyor** (no public dozer dataset exists). Its saved scores matched the Belt Conveyor's saved scores; a later Belt image rerun differed. This is a mapping, not separate evidence.
- Only the Pump reaches 90% on the main test. The reasons and caveats are in the [Week 6 summary](Documentation/week_06_professor_remarks.md).
- **More-than-80% in both modalities is currently met only by Pump (1 of 8 equipment mappings).** Wheel Loader, Drill Rig and Grader have no evaluated pair yet. Reusing the Belt Conveyor's CORA runs under another machine name does not create a new validation result; the [methodology](Documentation/methodology.md) defines the accuracy gate.
- XGBoost and LightGBM were compared with the random forest on the same tests and did not beat it ([Week 6 ensembles](Documentation/week_06_ensembles.md)).

**Week 7 addition — Wheel Loader (electric drive motor), machine 6.** New CORA fault data (half-broken and broken rotor bars, not used by any other machine), 4 classes: healthy, bearing defect, half-broken bar, broken bar. Training-data-only preprocessing. [Week 7 Wheel Loader](Documentation/week_07_wheel_loader.md).

| Machine (component) | Classes | Sensor | Images (CNN) | Sensor, time split | Images, time split |
|---|---|---:|---:|---:|---:|
| **Wheel Loader** (electric drive motor) | healthy, bearing, half-broken bar, broken bar | 0.312 | 0.312 | 0.938 | 0.812 |

Both branches are at chance (0.250) on the unseen-speed test, as for the Excavator.

## Quick start

You need [Python 3.11](https://www.python.org/downloads/) (tick **Add python.exe to PATH**), [VS Code](https://code.visualstudio.com/) and [Git](https://git-scm.com/downloads).

1. **Get the project.** In VS Code press `Ctrl+Shift+P` → **Git: Clone**, paste `https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment.git`, then open the folder.
2. **Set it up once.** In the terminal run `powershell -ExecutionPolicy Bypass -File .\setup.ps1` (Mac/Linux: `bash setup.sh`). Add `-Data` (`--data`) to fetch only the source files available automatically. The CORA vibration files and other publisher-gated inputs need the separate intake steps in the [dataset manifest](Dataset_Manifest/manifest.md).
3. **Run it.** Open a notebook in `Notebook/`, pick the **Capstone (.venv)** kernel at the top right and click **Run All**. Or press `Ctrl+Shift+B` to run all six model scripts after their required local data are present.

The notebooks still show the saved results when the data is missing. If you get `No module named cv2`, the wrong kernel is selected: choose **Capstone (.venv)**.

On macOS, if an ensemble run cannot load `xgboost` or `lightgbm` because `libomp.dylib` is missing, install the system OpenMP runtime or set `DYLD_LIBRARY_PATH` to the `torch/lib` directory inside this environment before launching Python. This is a local runtime issue, not evidence that the saved model results have been reproduced.

## Repository layout

```text
Models/          one .py file per machine: pump, haul_truck, belt_conveyor, excavator, dozer, wheel_loader
Notebook/        one short notebook per machine that runs its .py file and shows the results
Results/<machine>/   saved metrics (JSON) and written comparisons
Figures/<machine>/   result charts
Documentation/   weekly progress, methodology, full capstone details
Dataset_Manifest/    where every dataset comes from, its licence and limits
setup.ps1, setup.sh  one-step setup; requirements.txt / requirements-lock.txt
```

Local only (git-ignored): `Raw_Data/`, `Processed_Data/` (feature CSVs and saved models), `Images/` and `.venv/`.

## Main commands

Run these from the project folder (Windows: `.venv\Scripts\python.exe`, Mac/Linux: `.venv/bin/python`):

| Command | What it does |
|---|---|
| `python Models/<machine>.py` | list every command for that machine |
| `python Models/<machine>.py csv` | sensor (CSV) model, main test; add `--split time` for the time split |
| `python Models/<machine>.py export-images` | save the named images to `Images/<machine>/` |
| `python Models/<machine>.py cnn-pretrained` | pretrained ResNet18 with data augmentation |
| `python Models/<machine>.py baselines` | Week 5 baseline comparison (Pump, Haul Truck, Belt Conveyor) |
| `python Models/<machine>.py ensembles` | proposal Week 6: XGBoost and LightGBM vs random forest and others |
| `python Models/<machine>.py anomaly` | Week 7: anomaly detection trained on healthy data only (Isolation Forest, One-Class SVM) |
| `python Models/<machine>.py train-final` | train the sensor model on all data and save it |
| `python Models/<machine>.py predict --csv file.csv --label CLASS` | predict any CSV and show its accuracy (CORA machines also need `--speed F15`) |

## Documentation

- [Week 4 progress](Documentation/week_04_progress.md), [Week 5 baselines](Documentation/week_05_progress.md), [Week 6 professor's remarks](Documentation/week_06_professor_remarks.md), [Week 6 XGBoost/LightGBM](Documentation/week_06_ensembles.md), [Week 6 Dozer](Documentation/week_06_dozer.md), [Week 7 anomaly detection](Documentation/week_07_anomaly_detection.md), [Week 7 Wheel Loader](Documentation/week_07_wheel_loader.md)
- [Methodology](Documentation/methodology.md) and [capstone details](Documentation/capstone_details.md) (data download, reproduction, limits)
- [Dataset manifest](Dataset_Manifest/manifest.md) covering all eight planned equipment types, and the [collaboration guide](CONTRIBUTING.md)

These models classify the **current condition** of a component. They do not predict remaining useful life. Sensor and image results come from separately tested models; sensor and image data from unrelated sources are never paired.

Project code and documentation are under the [MIT License](LICENSE). Source datasets keep their publishers' terms, recorded in the dataset manifest, and are not redistributed here.
