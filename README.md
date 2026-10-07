# AI-Based Predictive Maintenance for Mining Equipment

Saskatchewan Polytechnic PROJ 613 capstone · Ganesh Mishra, Lakshya Sharma, and Sagar Patel · 12-week project

This repository has **one active dataset architecture**: component-level sensor and visual fault-condition monitoring for eight mining-equipment categories. The [capstone folder](Mining_Equipment_Condition_Monitoring_Capstone/README.md) contains the code, dataset verification matrix, methodology, results, and figures. Raw downloads and derived feature caches are local-only and are not committed.

## Current checkpoint: Week 4

| Status | Equipment |
|---|---|
| Sensor-versus-visual experiment completed | Pump support bearing; Belt Conveyor geared-drive analogue |
| Dataset/component pair mapped, experiment planned | Haul/Dump Truck, Excavator, Dozer, Wheel Loader, Drill Rig, Grader |

The [eight-equipment matrix](Mining_Equipment_Condition_Monitoring_Capstone/Dataset_Manifest/manifest.md) records source access, comparable labels, licences, and limits. Read the [methodology](Mining_Equipment_Condition_Monitoring_Capstone/Documentation/methodology.md) and [Week 4 progress checkpoint](Mining_Equipment_Condition_Monitoring_Capstone/Documentation/week_04_progress.md) before interpreting the scores.

- **Pump:** held-out sensor accuracy/macro F1 **0.958/0.958** and event-camera **1.000/1.000** on separate 24-recording tests. The data come from different physical rigs; the numbers are not a controlled modality ranking or mining-pump validation. [Full comparison](Mining_Equipment_Condition_Monitoring_Capstone/Results/pump_sensor_visual_comparison.md).
- **Belt Conveyor:** laboratory geared-drive T1 sensor **0.625/0.564** and infrared **0.500/0.418** on 16 run-level, held-out-speed predictions. Both accuracies fall below the **0.750 majority-class control**. This is a completed experiment with weak generalization, not an effective mine-conveyor detector. [Full comparison](Mining_Equipment_Condition_Monitoring_Capstone/Results/belt_conveyor_sensor_visual_comparison.md).

These models classify **current component condition**, not a future repair date or remaining useful life. No image and sensor observation from unrelated sources is falsely paired. The six planned equipment categories do not inherit completed results from the shared gearbox benchmark.

## Work locally in VS Code

Open this repository root in VS Code. Create a Python environment and install the [single dependency list](requirements.txt):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On Windows PowerShell, use `py -3.11 -m venv .venv` followed by `.venv\Scripts\python.exe -m pip install -r requirements.txt`. VS Code's default build task (`Cmd+Shift+B` on macOS; `Ctrl+Shift+B` on Windows) runs the **completed Week 4 Pump and Belt Conveyor analyses** in sequence. It uses the local `.venv` and does not fetch full source archives; the Pump visual script may fetch bounded remote event ranges if its cache is absent. Prepare the required local source files/caches first by following the [capstone reproduction instructions](Mining_Equipment_Condition_Monitoring_Capstone/README.md). The four individual tasks are also available through **Terminal → Run Task**. Do not mistake a missing local dataset for a model failure.

For detailed acquisition, evaluation and scientific limitations, use the capstone README. Large source archives, event recordings, raw and processed data, fitted models, virtual environments and caches must stay out of Git. The [collaboration guide](CONTRIBUTING.md) explains how to contribute without overstating results.

Original project code and documentation are under the [MIT License](LICENSE). Source datasets retain their publishers' separate terms, recorded in the dataset matrix; they are not redistributed here.
