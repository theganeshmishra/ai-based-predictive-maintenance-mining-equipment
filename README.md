# AI-Based Predictive Maintenance for Mining Equipment

Saskatchewan Polytechnic PROJ 613 capstone · Ganesh Mishra, Lakshya Sharma and Sagar Patel

For each machine we detect the **current fault condition of one component** two separate ways, from a **sensor** (vibration or temperature, as a CSV of features) and from a **camera** (CNN on images), and compare them. All data are public laboratory datasets used as stand-ins for mining equipment, so the results show the method, not a validated mine-site detector.

## Results (accuracy, 4 classes)

| Machine (component) | Classes | Sensor | Images (CNN) | Sensor, time split | Images, time split |
|---|---|---:|---:|---:|---:|
| **Pump** (support bearing) | healthy, inner, outer, ball | **0.958** | **1.000** | — | — |
| **Haul Truck** (wheel-motor bearing) | healthy, inner, outer, ball | 0.800–0.867 | 0.417 | 1.000 | 0.913 |
| **Belt Conveyor** (geared drive) | healthy, wear 25/50/75% | 0.625 | 0.562 | 1.000 | 1.000 |
| **Excavator** (swing drive) | healthy, bearing, unbalance, misalignment | 0.188 | 0.312 | 0.938 | 0.938 |

- **Main test (columns 3–4):** whole recordings, bearings or speeds the model never saw in training. This is the fair result.
- **Time split (columns 5–6):** the model trains on the first 60% of each recording and is tested on the last 30%. This test is easier because the same recordings appear in training, so it is reported separately.
- Only the Pump reaches 90% on the main test. The reasons and caveats are in the [Week 6 summary](Documentation/week_06_professor_remarks.md).

## Quick start

You need [Python 3.11](https://www.python.org/downloads/) (tick **Add python.exe to PATH**), [VS Code](https://code.visualstudio.com/) and [Git](https://git-scm.com/downloads).

1. **Get the project.** In VS Code press `Ctrl+Shift+P` → **Git: Clone**, paste `https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment.git`, then open the folder.
2. **Set it up once.** In the terminal run `powershell -ExecutionPolicy Bypass -File .\setup.ps1` (Mac/Linux: `bash setup.sh`). Add `-Data` (`--data`) to also download the data that can be fetched automatically (about 1.1 GB).
3. **Run it.** Open a notebook in `Notebook/`, pick the **Capstone (.venv)** kernel at the top right and click **Run All**. Or press `Ctrl+Shift+B` to run all 4 machines.

The notebooks still show the saved results when the data is missing. If you get `No module named cv2`, the wrong kernel is selected: choose **Capstone (.venv)**.

## Repository layout

```text
Models/          one .py file per machine: pump, haul_truck, belt_conveyor, excavator
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
| `python Models/<machine>.py train-final` | train the sensor model on all data and save it |
| `python Models/<machine>.py predict --csv file.csv --label CLASS` | predict any CSV and show its accuracy (CORA machines also need `--speed F15`) |

## Documentation

- [Week 4 progress](Documentation/week_04_progress.md), [Week 5 baselines](Documentation/week_05_progress.md), [Week 6 professor's remarks](Documentation/week_06_professor_remarks.md)
- [Methodology](Documentation/methodology.md) and [capstone details](Documentation/capstone_details.md) (data download, reproduction, limits)
- [Dataset manifest](Dataset_Manifest/manifest.md) covering all eight planned equipment types, and the [collaboration guide](CONTRIBUTING.md)

These models classify the **current condition** of a component. They do not predict remaining useful life. Sensor and image results come from separately tested models; sensor and image data from unrelated sources are never paired.

Project code and documentation are under the [MIT License](LICENSE). Source datasets keep their publishers' terms, recorded in the dataset manifest, and are not redistributed here.
