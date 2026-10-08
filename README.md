# AI-Based Predictive Maintenance for Mining Equipment

Saskatchewan Polytechnic PROJ 613 capstone · Ganesh Mishra, Lakshya Sharma, and Sagar Patel · 12-week project

This repository has **one active dataset architecture**: component-level sensor and visual fault-condition monitoring for eight mining-equipment categories. The code, dataset verification matrix, methodology, results, and figures all live at the top level of this repository; [capstone details](Documentation/capstone_details.md) has the full evidence and reproduction notes. Raw downloads and derived feature caches are local-only and are not committed.

## Repository layout

```text
Notebook/          one notebook per machine: Pump, Belt_Conveyor, Haul_Truck
capstone_code.py   all experiment and data-intake code in one file (backup; run: python capstone_code.py <command>)
Results/           committed metrics (JSON) and written comparisons
Figures/           committed result figures
Dataset_Manifest/  eight-equipment dataset verification matrix
Documentation/     methodology, Week 4 checkpoint, capstone details
Raw_Data/          local only, git-ignored: downloaded source data
Processed_Data/    local only, git-ignored: derived feature caches
```

## Current checkpoint: Week 4

| Status | Equipment |
|---|---|
| Sensor-versus-visual experiment completed | Pump support bearing; Belt Conveyor geared-drive analogue; Haul/Dump Truck wheel-motor bearing (Equipment #3, after Week 4) |
| Dataset/component pair mapped, experiment planned | Excavator, Dozer, Wheel Loader, Drill Rig, Grader |

The [eight-equipment matrix](Dataset_Manifest/manifest.md) records source access, comparable labels, licences, and limits. Read the [methodology](Documentation/methodology.md) and [Week 4 progress checkpoint](Documentation/week_04_progress.md) before interpreting the scores.

- **Pump:** held-out sensor accuracy/macro F1 **0.958/0.958** and event-camera **1.000/1.000** on separate 24-recording tests. The data come from different physical rigs; the numbers are not a controlled modality ranking or mining-pump validation. [Full comparison](Results/pump_sensor_visual_comparison.md).
- **Belt Conveyor:** laboratory geared-drive T1 sensor **0.625/0.564** and infrared **0.500/0.418** on 16 run-level, held-out-speed predictions. Both accuracies fall below the **0.750 majority-class control**. This is a completed experiment with weak generalization, not an effective mine-conveyor detector. [Full comparison](Results/belt_conveyor_sensor_visual_comparison.md).

- **Haul/Dump Truck (Equipment #3):** UODS vibration **0.900/0.871** on 20 recordings with whole bearings held out (most-frequent control 0.500); XJTU-Rotor event camera **0.444/0.356** on 9 recordings at an unseen speed (chance 0.333), and it never identifies inner-race faults. Different laboratory rigs and very small test sets; not haul-truck validation. [Full comparison](Results/haul_truck_sensor_visual_comparison.md).

These models classify **current component condition**, not a future repair date or remaining useful life. No image and sensor observation from unrelated sources is falsely paired. The five planned equipment categories do not inherit completed results from the shared gearbox benchmark.

## Work locally in VS Code

Open this repository root in VS Code. Create a Python environment and install the [single dependency list](requirements.txt):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On Windows PowerShell, use `py -3.11 -m venv .venv` followed by `.venv\Scripts\python.exe -m pip install -r requirements.txt`. VS Code's default build task (`Cmd+Shift+B` on macOS; `Ctrl+Shift+B` on Windows) runs the **completed Week 4 Pump and Belt Conveyor analyses** in sequence. It uses the local `.venv` and does not fetch full source archives; the Pump visual script may fetch bounded remote event ranges if its cache is absent. Prepare the required local source files/caches first by following the [capstone reproduction instructions](Documentation/capstone_details.md). The four individual tasks are also available through **Terminal → Run Task**. Do not mistake a missing local dataset for a model failure.

## Notebooks

The [`Notebook/`](Notebook/) folder has one notebook per completed machine, each holding both its sensor branch and its computer-vision branch:

- [`Pump.ipynb`](Notebook/Pump.ipynb): vibration vs event camera, plus the sensor-only pump baselines (original pump XLSX, UCI hydraulic, ESPset).
- [`Belt_Conveyor.ipynb`](Notebook/Belt_Conveyor.ipynb): T1 temperature vs infrared on the CORA geared-drive rig.
- [`Haul_Truck.ipynb`](Notebook/Haul_Truck.ipynb): UODS vibration vs XJTU-Rotor event camera for the electric wheel-motor bearing.

Each notebook is self-contained: all the code is inside it, in numbered steps (load data, build features, train and test, compare), with a plain-English note above each step. The last step checks that the notebook reproduces the saved results in `Results/`. `capstone_code.py` keeps all the original code in one file as a backup; it is what writes those saved results (for example `python capstone_code.py haul-truck`). Notebooks for the five planned machines will be added once their experiments are done. The PHM rock-drill baseline (`capstone_code.py rock-drill`) is sensor-only and belongs to the planned Drill Rig, so it has no notebook yet. To use them in VS Code, install the notebook kernel once with `.venv\Scripts\python.exe -m pip install ipykernel` and select `.venv` as the kernel. The notebooks read only local files in `Raw_Data/` and `Processed_Data/`, so they run offline once the data is downloaded.

For detailed acquisition, evaluation and scientific limitations, use [capstone details](Documentation/capstone_details.md). Large source archives, event recordings, raw and processed data, fitted models, virtual environments and caches must stay out of Git. The [collaboration guide](CONTRIBUTING.md) explains how to contribute without overstating results.

Original project code and documentation are under the [MIT License](LICENSE). Source datasets retain their publishers' separate terms, recorded in the dataset matrix; they are not redistributed here.
