# AI-Based Predictive Maintenance for Mining Equipment

Saskatchewan Polytechnic PROJ 613 capstone · Group 9 · Ganesh Mishra, Lakshya Sharma and Sagar Patel

This capstone demonstrates sensor and visual classification of current component condition across eight mining equipment mappings. The sources include laboratory analogues, small-gear inspection photos, and synthetic CAD renders. The eight mappings have working local prediction routes. The strength and comparability of the archived experiments vary by equipment.

## Results

The [eight-equipment evidence matrix](Documentation/equipment_evidence_matrix.md) records each component, fault label, source, test group, accuracy, balanced accuracy, and macro F1. The [results index](Results/README.md) explains how to read the saved score files.

- **Pump** has the strongest aligned held-out benchmark. Its sensor and event-camera results use different laboratory rigs and holdout protocols.
- **CORA** shares run IDs across sensor and infrared recordings. The four mappings scored 18.75%–56.25% when whole speeds were held out. The 93.75%–100% time-split scores measure later segments of the same recordings. **Dozer reuses all Belt Conveyor runs**, and Excavator and Wheel Loader share eight run IDs.
- **Drill Rig** uses CAD renders. The source labels CAD R0 as root breakage, while the sensor class describes a root crack. The full four-class comparison needs that qualification.
- **Motor Grader** reports separate results for slewing-bearing vibration and gear-tooth photo inspection. Its image headline is balanced accuracy per tooth crop.
- **Local deployment checks** verify routing and prediction output. They use saved deployment checkpoints and may include training samples. See the [deployment verification report](Documentation/deployment_verification.md) for the test cases.

The [20-day audit and plan](Documentation/proj613_20_day_audit.md) ranks the remaining evidence gaps and gives acceptance criteria. The [follow-up experiments](Documentation/followup_experiments.md) (November 6) carry out priorities 2 to 6:

- **CORA healthy vs worn** (Belt Conveyor and Dozer runs, unseen speed, 16 runs): sensor balanced accuracy 45.8% and images 75.0%, against 50.0% for the majority control. The image result beats the four-class model read the same way (58.3%) by two runs, which is promising but not defensible evidence (exact p = 0.05). No model is replaced, and CORA modelling stops here.
- **Haul Truck:** the ball class is confirmed load-confounded (no load in all 5 ball recordings, 400 N in the rest). Without it, the sensor model scores 84.0% accuracy (73.3% balanced) on 25 unseen-bearing recordings.
- **Drill Rig and Pump:** the crack mapping stays a qualified analogue; the Pump test is an unseen-speed-profile test, because the source gives no bearing IDs.
- **Grader:** the sensor and image experiments stay separate tasks.
- **Demo routes:** 15 of 16 routes rerun on 6 November with checkpoint hashes recorded (the Grader photo route needs its checkpoint on the demo computer).

Three [improvement attempts](Documentation/improvement_results.md), each [planned in advance](Documentation/improvement_plan_2026-11-06.md) and run once:

- **Drill Rig sensor with tachometer-order features:** 77.5% accuracy (75.7% balanced) on the same unseen-speed test, up from 73.3%. Recommended for retraining; the checkpoint is not replaced.
- **Haul Truck images with speed-scaled frames:** 50.0%, no gain over the archived 58.3%. The image model mostly recognises the camera angle.
- **CORA transient runs, unseen repetition at a trained speed:** 100% Belt Conveyor, 100% Excavator, 97.5% Wheel Loader. This is a narrower test than the fair test and does not replace it.

Under the fair test, Pump, Grader, Drill Rig (images) and Haul Truck (sensor accuracy, exactly 80%) reach 80%; the four CORA machines do not.

## How it works

The project uses one Python module and two notebooks.

- [Notebook/pdm.py](Notebook/pdm.py) holds all the code. Its sections run in workflow order, top to bottom.
- [Notebook/Mining_Equipment_Condition_Monitoring.ipynb](Notebook/Mining_Equipment_Condition_Monitoring.ipynb) walks through the workflow and shows the results.
- [Notebook/Integrated_Equipment_Fault_Prediction.ipynb](Notebook/Integrated_Equipment_Fault_Prediction.ipynb) accepts uploads and runs the local deployment checkpoints.

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

## Interactive fault classification

Open [Integrated Equipment Fault Prediction](Notebook/Integrated_Equipment_Fault_Prediction.ipynb) in VS Code, select the project’s `.venv` kernel and run all cells. Install `ipywidgets>=8,<9` in that environment if the upload widget is missing. Upload one supported CSV, MAT or image and select **Predict**. The notebook checks the input format and local provenance, then displays the equipment mapping, fault class, model, and confidence.

- **Automatic** uses recognized sensor schemas or an exact match to a trusted local image export. Keep the `Images/` indexes and frames locally for automatic image matching. The separate **Whole-machine photo** mode described below provides unverified visual suggestions.
- Shared CORA inputs and event-camera frames with unresolved provenance return **Ambiguous Equipment**. Choose a compatible mapping only when you know the source. The display identifies this as a user-confirmed laboratory mapping.
- Keep trusted deployment checkpoints in `Processed_Data/models/`. They contain the estimator or weights, the class order, and preprocessing metadata. The notebook displays availability and loads the models for inference.
- Motor Grader’s gear-tooth image experiment measures visible damage on a separate set of small gears. Its local deployment checkpoint is now built from the verified Roboflow v1 export. The integrated Grader sensor model classifies slewing-bearing conditions; gear-tooth predictions do not diagnose the slewing bearing.
- Drill Rig images come from synthetic CAD gear renders. The Drill Rig and Grader sensor deployment fits use two-second excerpts per publisher recording. The Drill Rig image checkpoint uses a bounded mesh subset. Those deployment fits are separate from the published experiments.

The November 5 post-push smoke test ran all eight sensor routes and all eight image routes and passed 19 upload and error checks. The local record is `Processed_Data/pushed_deployment_verification.json`.

For raw sensor CSV uploads, use these schemas. The header identifies the expected source format and rate. Prepare the CSV from the corresponding recording without resampling or substituting channels.

| Source mapping | CSV columns and requirements |
|---|---|
| Pump, Ottawa | `ottawa_accel_200khz` from MAT `Channel_1`. At least 20,000 samples at 200 kHz. |
| Haul Truck, UODS | `uods_accel_42khz,uods_mic_42khz` from the first two MAT columns. At least 21,000 samples at 42 kHz. The upload uses complete half-second blocks. |
| CORA mappings | Original one-column, headerless Vx CSV with the published speed in the filename. Confirm the laboratory mapping when prompted. |
| Drill Rig, MCC5-THU | Original motor and gearbox vibration channel headers at 12.8 kHz. |
| Motor Grader, JUST | Original six accelerometer channel headers at 50 kHz. |

See the [deployment verification report](Documentation/deployment_verification.md) for the per-equipment results and demonstration limits.

The fair tests above hold out complete recordings, speeds, bearings or designs. The time-split experiments use later segments of recordings also seen in training. Inference smoke tests check routing and output on known local samples. Model confidence is a class probability, and the task is current laboratory component condition classification.

### Motor Grader gear-tooth deployment

The saved 82.6% balanced accuracy comes from the separate gear-tooth experiment with held-out source groups. The local `Processed_Data/models/grader_gear_tooth.joblib` checkpoint was trained from the Roboflow v1 COCO export and verified through all three interfaces on labeled example crops. To rebuild it, place the verified export at `Raw_Data/roboflow_gear_photos/GearBox_adddingImg.v1i.coco.zip` and run `python Notebook/pdm.py grader train-final-images`. The command verifies the archive SHA-256 and fits the fixed ResNet18 feature extractor plus logistic regression configuration.

Select **Motor Grader** in the notebook or **Grader** in the local upload page, or run `python Notebook/pdm.py predict tooth_crop.jpg --machine grader`. Supply a manually verified crop of one gear tooth with a small margin. The output classifies visible gear-tooth condition. The Grader sensor model covers slewing-bearing conditions separately. The gear-tooth route reports model unavailable when the fitted checkpoint is absent.

### Whole-machine photo identification

Choose **Whole-machine photo** in the task menu and upload a site photo. The optional BuildVision v0.3 detector supports excavators, wheel loaders and dump trucks. A single detection returns **Unknown Equipment** with an unverified suggestion, a box in image pixels, and a detector score. Several supported categories return **Ambiguous Equipment**. With no detections above the threshold, the notebook returns **Unknown Equipment** without a suggestion. A labeled Grader example produced a Wheel Loader candidate at 86.4%, and a dump-truck detection does not establish mining haul-truck provenance. Use **Component fault** for supported sensor files and laboratory component images.

Install `onnxruntime` in the notebook environment and put the publisher’s [ONNX model and config](https://huggingface.co/constructelligence/construction-site-safety-hazards/tree/79673cff5a8bb249fc8ec9cca3daa19e2ef95191) in `Processed_Data/models/whole_machine/`. The optional model is published under AGPL-3.0 and runs locally. Photo mode never routes detections to a fault model. Motor Grader slewing-bearing image diagnosis still requires appropriate labeled data.

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
| `python Notebook/pdm.py <machine> train-final-images` | save a supported ResNet18 deployment checkpoint from exported frames | minutes to hours |
| `python Notebook/pdm.py predict <file>` | identify the file's dataset format, then the machine's model gives the condition; shared formats return ambiguity unless a mapping is supplied | seconds |
| `python Notebook/pdm.py serve` | upload page at http://localhost:8613 (this computer only): drop in a file, get machine, condition and confidence | – |
| `python Notebook/pdm.py cora-binary verify\|sensor\|images` | CORA healthy vs worn follow-up → `Results/experiments/cora_binary/` | 1 minute / 1 minute / about 2 hours |
| `python Notebook/pdm.py haul-truck-load-check` | Haul Truck sensor fair test without the unloaded ball class | about 15 minutes |
| `python Notebook/pdm.py attempt drill-orders\|haul-frames\|cora-transient` | the three planned improvement attempts → `Results/experiments/improvement/` (cora-transient needs `Raw_Data/cora_transient/`) | minutes / hours on CPU / minutes |
| `python Notebook/pdm.py route-check` | run every predict route once, record checkpoint SHA-256 → `Results/deployment/route_check.json` | about 1 minute |
| `python Notebook/pdm.py data-check` | list missing raw files | seconds |

Machines: `pump`, `haul_truck`, `belt_conveyor`, `excavator`, `dozer`, `wheel_loader`, `drill_rig`, `grader`.

`predict` first checks which dataset format the file matches (columns, variable names, image size and colours). This is a format check, not machine learning. Shared formats return **Ambiguous Equipment**. Use `--machine` when the source mapping is known. The notebook, command line, and upload server use the same source checks. Drill Rig renders must match the trusted local export index. Blank images are rejected before fault inference.

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

- The fair-test image scores of the first seven machines were produced before this code was reorganised (commit c7d8d25) and carried over, because each run takes hours to a day on a CPU. The reorganised code gave identical results on a Dozer test subset (single CPU thread). On a Drill Rig subset it gave the same per-gear accuracy (0.271) but chose different epochs in one round, so the gear-image path is not confirmed identical. The Grader images and all image time splits were run with the reorganised code. PyTorch training on a multi-core CPU is not bit-for-bit repeatable, so a rerun can differ slightly.
- **GPU (optional).** `requirements.txt` installs the CPU build of PyTorch, which works everywhere. With an NVIDIA GPU, run `.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130`. The image model then uses the GPU by itself (set `PDM_DEVICE=cpu` to force the CPU). Each image result records the device it was trained on, because GPU and CPU runs can differ slightly.
- These models classify the current condition of a component. They do not predict remaining useful life.
- Project code and documentation are under the [MIT License](LICENSE). Datasets keep their publishers' licences (all CC BY 4.0) and are not redistributed here.
