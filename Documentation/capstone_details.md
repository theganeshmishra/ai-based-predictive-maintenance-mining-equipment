# Mining equipment condition monitoring capstone

Saskatchewan Polytechnic AIDA / PROJ 613. Working title: **Machine Learning-Based Condition Monitoring of Mining Equipment Using Multi-Sensor Data and Computer Vision**.

This capstone remains an eight-equipment research project, **not** a claim that eight validated mining-machine fault models exist. Source files and derived training tables live in ignored local folders; the small manifest, code, results, and figures are suitable for review. The [folder-specific ignore rules](../.gitignore) keep source downloads, raw images/events, caches, and model binaries out of GitHub while allowing the reviewed figures.

Start with the [eight-category dataset matrix](../Dataset_Manifest/manifest.md),
[cross-equipment methodology](methodology.md),
[completed Pump comparison](../Results/pump/pump_sensor_visual_comparison.md), and
[Belt Conveyor geared-drive comparison](../Results/belt_conveyor/belt_conveyor_sensor_visual_comparison.md),
and [Week 4 progress checkpoint](week_04_progress.md). The
equipment folders are currently empty local placeholders; the matrix and
methodology preserve all eight categories without inventing results for them.

## Week 4 checkpoint (October 5, 2026)

The eight equipment categories and component/fault dataset matrix are mapped.
Only **Pump** and **Belt Conveyor** have completed sensor-versus-image fault
experiments. The other six categories—Haul/Dump Truck, Excavator, Dozer, Wheel
Loader, Drill Rig, and Grader—are **planned, not implemented** as matched
sensor/image experiments. No Equipment #3 modelling is part of Week 4.

**After Week 4:** the Haul/Dump Truck (electric wheel-motor bearing) was added
as Equipment #3; see [its comparison](../Results/haul_truck/haul_truck_sensor_visual_comparison.md)
and `Notebook/Haul_Truck.ipynb`. Five categories remain planned.

### Current evidence

| Experiment | Source domain | Task and evaluation | Held-out result | What it does not show |
|---|---|---|---|---|
| UCI hydraulic system | Controlled hydraulic test rig | Three-class current pump-leakage severity; complete condition-state combinations held out | Random forest: macro F1 **0.963**, balanced accuracy **0.963** on 290 stable cycles | Mining-vehicle performance or a future repair date |
| ESPset V3 | Real offshore oil-production pumps | Normal vs abnormal vibration condition; physical pump IDs held out | Random forest: macro F1 **0.914**, abnormal recall **0.837** on 1,503 tests from three pumps | Mining-pump performance, RUL, or image/sensor fusion |

These scores are **not directly comparable**: the machines, labels, class balance, and test units differ. The source/quality checks, split IDs, candidate-model validation, confusion matrices, and limitations are in [`Results/`](../Results/). The images in [`Figures/`](../Figures/) are diagnostic plots, not equipment photographs.

One comparison that *is* made on the same data and split:

- **Pump baseline value:** on held-out ESPset pumps, the selected forest scored 0.914 macro F1 and 0.837 abnormal recall, versus 0.428 macro F1 and zero abnormal recall for a most-frequent baseline.

The two rows above are **sensor-only** findings. The Pump support-bearing experiment now has separately tested sensor and visual branches on the same four current-fault classes. The University of Ottawa bearing-rig sensor proxy scored **0.958 accuracy and macro F1 on 24 recording-held-out runs**; the XJTU Pump event-camera/OpenCV branch scored **1.000 accuracy and macro F1 on 24 recording-held-out runs**. Secondary held-out-operating/camera checks scored 0.958 and 0.855 macro F1, respectively. These are **not** a controlled head-to-head modality comparison: sources and operating shifts differ. The original centrifugal-pump XLSX model remains a same-recording diagnostic and is excluded from the independent comparison. See the [full Pump metrics, per-class results, confusion matrices and limitations](../Results/pump/pump_sensor_visual_comparison.md).

The Belt Conveyor geared-drive analogue now has a **same-CORA-run sensor/infrared comparison** across 16 stationary experiments, with each of four speeds held out in turn. The T1-temperature model reached **0.625 accuracy / 0.564 macro F1**; the infrared/OpenCV model reached **0.500 / 0.418**. Both were below the **0.750 majority-class accuracy** control. This is a completed, leakage-aware component-level experiment, **not** an effective fault detector or a mining-conveyor validation. See the [full run-level comparison](../Results/belt_conveyor/belt_conveyor_sensor_visual_comparison.md).

## Eight-category map

| Category | Selected comparable component target | Verification grade |
|---|---|---|
| Haul / dump truck | Electric-drive wheel-motor bearing: UODS vibration/acoustic vs XJTU event camera | YELLOW — healthy/inner/outer labels align; no truck recordings; exclude UODS ball class because load differs |
| Excavator | Swing-drive gearbox gear wear: CORA sensor and infrared | YELLOW — shared lab gearbox analogue |
| Dozer | Final-drive gear wear: CORA sensor and infrared | YELLOW — same shared analogue |
| Wheel loader | Final-drive/transmission gear wear: CORA sensor and infrared | YELLOW — same shared analogue |
| Drill rig | Rotary-head reduction gear wear: CORA sensor and infrared | YELLOW — **not** the PHM rock-drill pressure faults |
| Grader | Circle-drive gear wear: CORA sensor and infrared | YELLOW — lab gear to circle-drive transfer is limited |
| Belt conveyor | **Geared** drive reducer wear: CORA sensor and infrared | YELLOW — not belt cracks/idler health or gearless conveyors |
| Pump | Pump support-bearing faults: original centrifugal-pump vibration/XJTU event camera source pair; Ottawa bearing-rig proxy for repeat-run sensor evaluation | GREEN original source-label match; **YELLOW empirical transfer** from generic bearing rig to pump |

See the [eight-row verification table and dataset manifest](../Dataset_Manifest/manifest.md) for exact sources, counts, licences, matching labels, OEM component justification, and confidence. The **locked research architecture has only three component targets**, not eight independent machine datasets: CORA gear wear is reused across six categories; UODS/XJTU-Rotor serves electric-motor bearing; centrifugal-pump/XJTU-Pump serves pump bearing. The Ottawa bearing dataset is an additional **Pump sensor-evaluation proxy**, not a fourth equipment category or pump recording. CORA is same-rig sensor-plus-infrared at experiment level, pending timestamp validation for frame-level analysis. The other pairs are different physical rigs and are comparable **only by component/fault taxonomy**. No unrelated sensor row and image is joined. These experiments classify present condition, not time-to-repair or remaining useful life.

The UCI hydraulic benchmark remains useful **sensor-only** work, not a completed paired experiment for Excavator, Dozer, Wheel Loader or Grader. An earlier PHM 2022 rock-drill pressure benchmark was removed from the repository on October 7, 2026: it has no camera counterpart and a different fault target from the Drill Rig gear-wear mapping (it remains in Git history). Internal hydraulic pump leakage, valve lag, accumulator faults, and drill hydraulic faults still have **RED visual coverage** and are not claimed as solved by a gearbox or bearing image dataset. Equipment-identification images also remain out of the fault comparison.

The shared CORA gear/infrared source for six category mappings has bounded
file-inventory (`inspect-file`) and sensor/thermal preprocessing
(`preprocess`) helpers in [`Models/belt_conveyor.py`](../Models/belt_conveyor.py).
Its `intake` and `experiment` commands use 16 selected runs;
they do not train six equipment-specific models or infer sample-level
sensor/image alignment. From the repository root, run
`python Models/belt_conveyor.py inspect-file /path/to/Information_about_database_files.txt`
to inspect a locally held publisher file guide, or run
`python Models/belt_conveyor.py preprocess --thermal /path/to/BD_F15_S.mat`
for a bounded thermal smoke test. To reproduce Equipment #2 after installing
the requirements below, run:

```bash
python Models/belt_conveyor.py intake
python Models/belt_conveyor.py experiment
```

This intake requests only the 16 specified thermal MAT stacks (about 391 MB)
and bounded byte ranges of their 16 original T1 CSVs (about 34 MB requested),
not the complete 38.8 GB release. It writes the ignored local sensor cache,
the [file/provenance record](../Results/belt_conveyor/belt_conveyor_cora_intake.json),
[metrics](../Results/belt_conveyor/belt_conveyor_cora.json), and
[confusion matrices](../Figures/belt_conveyor/belt_conveyor_cora_confusions.png). Raw CORA files
remain ignored.

## Reproduce the first two baselines

Use the repository's single [dependency list](../requirements.txt). From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1` (or call `.venv\Scripts\python.exe` directly). Obtain these **small official archives** and leave them under ignored `Raw_Data/`:

1. [UCI hydraulic system ZIP](https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip) at `Raw_Data/uci_hydraulic/condition_monitoring_of_hydraulic_systems.zip` (73 MB; SHA-256 `24128aad2ee45eea7e6b63ebbd9992cdf25d0483a2cebefbfc13bc69079af1f2`).
2. [ESPset V3 `features.zip`](https://data.mendeley.com/datasets/m268jsw339/3) at `Raw_Data/espset_v3/features.zip` (258 KB; SHA-256 `b8214c10ba5b5b211c00a058f2fffa3ed7ccc46f67219b4769c907baec9a724d`). Only the published feature file is required for the initial benchmark; the 110 MB `spectrum.zip` was not downloaded.

Then run:

```bash
python Models/pump.py uci-hydraulic
python Models/pump.py espset
```

The scripts verify archive hashes, inspect source shapes and missing/non-finite values, build processed features locally, use group-aware splits, fit baseline models, and save JSON metrics plus diagnostic figures. If a source archive changes, the checksum check intentionally stops the run for review.

## Pump component-condition experiment

The [dataset matrix](../Dataset_Manifest/manifest.md) retains the original Pump-to-Pump source-label match. For an **independent-recording sensor evaluation**, this experiment uses the [University of Ottawa V2 bearing-vibration data](https://data.mendeley.com/datasets/v43hmbwxpm/2) as a mechanically comparable support-bearing proxy: healthy, inner race, outer race and ball fault, with 12 separate recordings per class. It is **not data collected from a pump or mine**. The original four centrifugal-pump XLSX files remain an explicitly labelled within-recording diagnostic. The **16.49 GB XJTU event-camera ZIPs were not downloaded**: the first two seconds of the 48 selected recordings were read remotely and cached as compact features. All raw Ottawa files and processed features stay in Git-ignored local folders.

Run from the repository root:

```bash
python Models/pump.py intake
python Models/pump.py sensor
python Models/pump.py camera
python Models/pump.py compare
```

The intake script downloads only the 48 selected Ottawa MAT files (about **631 MB**) and checks each against the publisher's SHA-256; its [file manifest](../Results/pump/pump_uottawa_intake.json) records source identifiers and hashes. The sensor model uses 0.1-second non-overlapping vibration windows, trains on 24 complete A/B speed-profile recordings and tests on 24 disjoint C/D recordings, with one averaged prediction per recording. Its [result](../Results/pump/pump_bearing_independent.json) includes Accuracy, macro Precision/Recall/F1, per-class metrics, a most-frequent control, shuffled-run-label control, and held-out-profile folds. The visual script converts event streams into 250 ms event-count images, extracts OpenCV HOG features, trains on 20 Hz recordings and tests on separate 30 Hz recordings; its existing cached result is unchanged. The comparison script writes the [readable report](../Results/pump/pump_sensor_visual_comparison.md) and [confusion-matrix figure](../Figures/pump/pump_sensor_visual_confusions.png).

The original [centrifugal-pump vibration script](../Models/pump.py) is retained solely for provenance and can be run if its four source XLSX files are placed in ignored `Raw_Data/pump_bearing/`; its same-file score is **not** used in the independent comparison. Both main branches now use whole-run test units, but their different rigs and operating shifts still prohibit a numerical claim that one modality is better. There is no row-level sensor/image pairing or future-repair prediction. Within the professor's stated allowance for **sufficiently similar components on different machines**, this is a completed four-class, component-level sensor-plus-visual comparison—not mining-pump validation.
