# AI-Based Predictive Maintenance for Mining Equipment

Saskatchewan Polytechnic PROJ 613 capstone · Group 9 · Ganesh Mishra, Lakshya Sharma and Sagar Patel

For eight mining machines we pick one component and classify its current condition (four classes) two separate ways: from a **sensor** recording and from **camera images**. Then we compare the two. All data are public laboratory datasets used as stand-ins for mining equipment, and a machine's sensor and image sides come from different rigs. The results show the method; they are not a mine-site result.

## Results

Accuracy per test recording (per gear for the Drill Rig images; balanced accuracy per tooth crop for the Grader images). The **fair test** holds out whole recordings, speeds or bearings that the model never saw. The **time split** trains on the start of each recording and tests on its end, which is much easier.

| Machine | Component | Sensor, fair | Majority control | Images, fair | Sensor, time split | Images, time split |
|---|---|---|---|---|---|---|
| Pump | support bearing | 0.958 | 0.250 | 1.000 | – | – |
| Haul Truck | wheel-motor bearing | 0.800 | 0.500 | 0.583 | 1.000 | 1.000 |
| Belt Conveyor | geared drive | 0.438 | 0.250 | 0.500 | 1.000 | running |
| Excavator | drive train | 0.250 | 0.250 | 0.250 | 0.938 | running |
| Dozer | final drive | 0.438 | 0.250 | 0.562 | 1.000 | 0.938 |
| Wheel Loader | electric drive | 0.312 | 0.250 | 0.188 | 0.938 | running |
| Drill Rig | rotary-head reduction gear | 0.733 | 0.300 | 0.889 | – | – |
| Grader | circle-drive slewing bearing | 0.842 | 0.250 | 0.826 | – | – |

- "–" means that test does not apply (the Pump, Drill Rig and Grader data have no matching recordings to cut in time). "running" means that result was still being computed at this commit.
- Guessing gives about 0.25 (four classes). The majority control always predicts the most common training class.
- Only the Pump scores above 0.9 on both sides of the fair test.
- The four CORA machines score 0.94 to 1.00 on the sensor time split but only 0.19 to 0.56 on the fair test. The models recognise recordings they have already seen, but do not carry over to an unseen speed. There is only one run per class and speed.
- The Dozer reuses the Belt Conveyor's runs, so it is not independent evidence.
- The Drill Rig images are noise-free CAD renders, so 0.889 is an upper bound for a camera.
- The Grader image side is a separate two-class task (good vs damaged tooth crops of real gear photos, not the slewing ring), scored by balanced accuracy because 94% of the crops are good teeth.
- Test sets are small: on a 16-run test, one recording moves accuracy by 6 points.

## How it works

The project is **one Python file and one notebook**:

- [Notebook/pdm.py](Notebook/pdm.py) holds all the code. Its sections run in workflow order, top to bottom.
- [Notebook/Mining_Equipment_Condition_Monitoring.ipynb](Notebook/Mining_Equipment_Condition_Monitoring.ipynb) walks through the workflow and shows the results.

```text
Raw_Data/ ─► 1a load sensor data ─► one feature row per time window ─► 3b sensor model ─┐
          └► 1b load camera data ─► PNG images + index.csv          ─► 3c image model  ─┤
                                                                                        ▼
     2  split: fair test and time split               3a  one answer per recording
     4  MACHINES: each machine's data and what its fair test holds out
```

| Section of `pdm.py` | What it does |
|---|---|
| 0 | project folders |
| 1 | load data: raw sensor files → feature rows (1a); raw camera files → PNG images (1b); data check (1c) |
| 2 | the fair test and the time split |
| 3 | one answer per recording and metrics (3a); sensor model (3b); ImageNet ResNet-18 image model (3c) |
| 4 | the eight machines: datasets, classes, what each fair test holds out |
| 5 | run the steps for one machine; writes `Results/` and `Figures/` |
| 6 | classify a new file |
| 7 | tables and figures for the notebook |
| 8 | command line |

The full method is in [Documentation/methodology.md](Documentation/methodology.md).

## Quick start

You need [Python 3.11](https://www.python.org/downloads/) (tick **Add python.exe to PATH**), [VS Code](https://code.visualstudio.com/) and [Git](https://git-scm.com/downloads).

1. Clone this repository and open the folder in a terminal.
2. Create the environment and install the packages (Mac/Linux: `python3` and `.venv/bin/python`):

   ```bash
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Download the data by hand from the sources in the [dataset manifest](Dataset_Manifest/manifest.md) into `Raw_Data/`, then check it:

   ```bash
   .venv\Scripts\python.exe Notebook\pdm.py data-check
   ```

4. Open [Notebook/Mining_Equipment_Condition_Monitoring.ipynb](Notebook/Mining_Equipment_Condition_Monitoring.ipynb) in VS Code, choose the `.venv` Python as the kernel and click **Run All**. The results section works without the raw data.

## Commands

From the project folder (Windows: `.venv\Scripts\python.exe`, Mac/Linux: `.venv/bin/python`):

| Command | What it does | Time on a laptop CPU |
|---|---|---|
| `python Notebook/pdm.py <machine> sensor` | sensor model, fair test → `Results/<machine>/sensor.json` | seconds to minutes |
| `python Notebook/pdm.py <machine> sensor --time-split` | sensor model, time split | seconds |
| `python Notebook/pdm.py <machine> export-images` | raw camera files → `Images/<machine>/` | minutes (Drill Rig: hours) |
| `python Notebook/pdm.py <machine> images` | image model, fair test → `Results/<machine>/images.json` | hours |
| `python Notebook/pdm.py <machine> images --time-split` | image model, time split | minutes to an hour |
| `python Notebook/pdm.py <machine> train-final` | sensor model on all data, for `predict` | minutes |
| `python Notebook/pdm.py predict <file>` | identify the file's dataset format, then the machine's model gives the condition; if several machines share the format, the condition under each | seconds |
| `python Notebook/pdm.py serve` | upload page at http://localhost:8613 (this computer only): drop in a file, get machine, condition and confidence | – |
| `python Notebook/pdm.py data-check` | list missing raw files | seconds |

Machines: `pump`, `haul_truck`, `belt_conveyor`, `excavator`, `dozer`, `wheel_loader`, `drill_rig`, `grader`.

`predict` first checks which dataset format the file matches (columns, variable names, image size and colours). This is a format check, not machine learning. Several machines share one laboratory rig, so for those formats the machine must be named with `--machine`.

## Folders

```text
Notebook/            pdm.py (all code) and the notebook: load → split → train → results → predict
Results/<machine>/   sensor.json, images.json and the time-split versions
Figures/<machine>/   confusion matrices for each result
Dataset_Manifest/    data sources, licences and the file lists with SHA-256
Documentation/       methodology
```

Local only (git-ignored): `Raw_Data/`, `Processed_Data/` (feature tables and trained models), `Images/` and `.venv/`.

## Notes

- The fair-test image scores were produced before this code was reorganised (commit c7d8d25) and carried over, because each run takes hours to a day on a CPU. The reorganised image code uses the same method. PyTorch training on a multi-core CPU is not bit-for-bit repeatable, so a rerun can differ slightly.
- **GPU (optional).** `requirements.txt` installs the CPU build of PyTorch, which works everywhere. With an NVIDIA GPU, run `.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130`. The image model then uses the GPU by itself (set `PDM_DEVICE=cpu` to force the CPU). Each image result records the device it was trained on, because GPU and CPU runs can differ slightly.
- These models classify the current condition of a component. They do not predict remaining useful life.
- Project code and documentation are under the [MIT License](LICENSE). Datasets keep their publishers' licences (all CC BY 4.0) and are not redistributed here.
