# AI-Based Predictive Maintenance for Mining Equipment

Saskatchewan Polytechnic PROJ 613 capstone · Ganesh Mishra, Lakshya Sharma, and Sagar Patel · 12-week project

This repository has **one active dataset architecture**: component-level sensor and visual fault-condition monitoring for eight mining-equipment categories. The code, dataset verification matrix, methodology, results, and figures all live at the top level of this repository; [capstone details](Documentation/capstone_details.md) has the full evidence and reproduction notes. Raw downloads and derived feature caches are local-only and are not committed.

## Quick start (5 minutes)

You need [Python 3.11](https://www.python.org/downloads/) (tick **Add python.exe to PATH** when installing), [VS Code](https://code.visualstudio.com/) and [Git](https://git-scm.com/downloads).

1. **Get the project:** in VS Code, press `Ctrl+Shift+P` → **Git: Clone** → paste `https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment.git` → open the folder.
2. **Set it up:** open the VS Code terminal (**Terminal → New Terminal**) and run one line:
   - Windows: `powershell -ExecutionPolicy Bypass -File .\setup.ps1`
   - Mac/Linux: `bash setup.sh`

   This creates `.venv`, installs everything, and adds the **Capstone (.venv)** notebook kernel.
3. **Run it:** open `Notebook/Pump.ipynb`, `Belt_Conveyor.ipynb` or `Haul_Truck.ipynb`, check the top right says **Capstone (.venv)** (click it to pick it if not), then click **Run All**. Or press `Ctrl+Shift+B` (`Cmd+Shift+B` on Mac) to run all 3 machines.

Without data the notebooks show the saved results. To download the data (about 1.1 GB) and rerun the models, run the setup again with `-Data` (Windows: `powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Data`; Mac/Linux: `bash setup.sh --data`).

**If something goes wrong:** `No module named cv2` means the notebook is using the wrong Python: click the kernel name at the top right → **Select Another Kernel → Jupyter Kernel → Capstone (.venv)**. If setup says Python was not found, install Python 3.11 with **Add to PATH** ticked and run it again.

## Repository layout

```text
Notebook/          one notebook per machine: Pump, Belt_Conveyor, Haul_Truck, Excavator
Models/            one .py file per equipment: pump.py, belt_conveyor.py, haul_truck.py, excavator.py
Results/           committed metrics (JSON) and written comparisons
Figures/           committed result figures
Dataset_Manifest/  eight-equipment dataset verification matrix
Documentation/     methodology, Week 4 and Week 5 progress, capstone details
requirements-lock.txt  exact package versions that reproduce the saved results
Raw_Data/          local only, git-ignored: downloaded source data
Processed_Data/    local only, git-ignored: derived feature caches
```

## Current checkpoint: Week 4

| Status | Equipment |
|---|---|
| Sensor-versus-visual experiment completed | Pump support bearing; Belt Conveyor geared-drive analogue; Haul/Dump Truck wheel-motor bearing (Equipment #3); Excavator swing-drive motor/gearbox faults (Equipment #4, CORA healthy / bearing defect / unbalance / misalignment) |
| Dataset/component pair mapped, experiment planned | Dozer, Wheel Loader, Drill Rig, Grader |

The [eight-equipment matrix](Dataset_Manifest/manifest.md) records source access, comparable labels, licences, and limits. Read the [methodology](Documentation/methodology.md) and [Week 4 progress checkpoint](Documentation/week_04_progress.md) before interpreting the scores.

- **Pump:** held-out sensor accuracy/macro F1 **0.958/0.958** and event-camera **1.000/1.000** on separate 24-recording tests. The data come from different physical rigs; the numbers are not a controlled modality ranking or mining-pump validation. [Full comparison](Results/pump_sensor_visual_comparison.md).
- **Belt Conveyor:** laboratory geared-drive T1 sensor **0.625/0.564** and infrared **0.500/0.418** on 16 run-level, held-out-speed predictions. Both accuracies fall below the **0.750 majority-class control**. This is a completed experiment with weak generalization, not an effective mine-conveyor detector. [Full comparison](Results/belt_conveyor_sensor_visual_comparison.md).

- **Haul/Dump Truck (Equipment #3):** UODS vibration **0.900/0.871** on 20 recordings with whole bearings held out (most-frequent control 0.500); XJTU-Rotor event camera **0.444/0.356** on 9 recordings at an unseen speed (chance 0.333), and it never identifies inner-race faults. Different laboratory rigs and very small test sets; not haul-truck validation. [Full comparison](Results/haul_truck_sensor_visual_comparison.md).

- **Week 5 baseline comparison:** majority-class control, logistic regression, decision tree and random forest on the same held-out tests (`python Models/<machine>.py baselines`). Pump models score 0.875–1.000 in both branches; no Belt Conveyor model beats the 0.750 majority control; Haul Truck vibration models reach 0.750–0.900 while every event-camera model stays near chance. Settings were fixed in advance, so the main models are unchanged. [Week 5 summary](Documentation/week_05_progress.md).

- **Week 6 (professor's remarks):** 4 classes for every machine, CSV sensor models, named images, CNNs with data augmentation and a pretrained ResNet18. Pump reaches 90%+ on both sensor (0.958) and images (1.000). Haul Truck 0.867 / 0.417, Belt Conveyor 0.625 / 0.562 and Excavator 0.188 / 0.312 under the strict tests; the Excavator reaches 0.938 / 0.938 on a separately labelled, easier time-split test. [Week 6 summary](Documentation/week_06_professor_remarks.md).

These models classify **current component condition**, not a future repair date or remaining useful life. No image and sensor observation from unrelated sources is falsely paired. The four planned equipment categories do not inherit completed results from the shared gearbox benchmark.

## Work locally in VS Code

The setup scripts above do the environment set-up; to do it by hand instead, create `.venv`, install the [single dependency list](requirements.txt), and register the kernel with `python -m ipykernel install --user --name capstone-week4 --display-name "Capstone (.venv)"` using the `.venv` Python. VS Code's default build task (`Cmd+Shift+B` on macOS; `Ctrl+Shift+B` on Windows), **Run all 3 machines**, runs the completed Pump, Belt Conveyor and Haul Truck experiments in sequence. It uses the local `.venv` and does not fetch full source archives; the Pump visual script may fetch bounded remote event ranges if its cache is absent. Prepare the required local source files/caches first by following the [capstone reproduction instructions](Documentation/capstone_details.md). **Run Pump**, **Run Belt Conveyor** and **Run Haul Truck** are also available separately through **Terminal → Run Task**. Do not mistake a missing local dataset for a model failure.

## Notebooks

The [`Notebook/`](Notebook/) folder has one notebook per completed machine, each holding both its sensor branch and its computer-vision branch:

- [`Pump.ipynb`](Notebook/Pump.ipynb): vibration vs event camera, plus the sensor-only pump baselines (original pump XLSX, UCI hydraulic, ESPset).
- [`Belt_Conveyor.ipynb`](Notebook/Belt_Conveyor.ipynb): T1 temperature vs infrared on the CORA geared-drive rig.
- [`Haul_Truck.ipynb`](Notebook/Haul_Truck.ipynb): UODS vibration vs XJTU-Rotor event camera for the electric wheel-motor bearing.
- [`Excavator.ipynb`](Notebook/Excavator.ipynb): CORA vibration vs infrared for drive-train faults (bearing defect, unbalance, misalignment).

Each notebook is short: it lists which of its datasets are present, then runs the project's scripts in `Models/` unchanged and shows the results, so notebooks and code cannot drift apart. The shared notebook set-up (finding the project, checking data, running scripts, showing tables) is in [`Notebook/notebook_helpers.py`](Notebook/notebook_helpers.py). Notebooks for the four planned machines will be added once their experiments are done. To use them in VS Code, select the **Capstone (.venv)** kernel at the top right of the notebook (the set-up cell warns if a different Python is in use). When all data is local, the notebooks block network access, so a clean run proves the work is reproducible offline.

For detailed acquisition, evaluation and scientific limitations, use [capstone details](Documentation/capstone_details.md). Large source archives, event recordings, raw and processed data, fitted models, virtual environments and caches must stay out of Git. The [collaboration guide](CONTRIBUTING.md) explains how to contribute without overstating results.

Original project code and documentation are under the [MIT License](LICENSE). Source datasets retain their publishers' separate terms, recorded in the dataset matrix; they are not redistributed here.
