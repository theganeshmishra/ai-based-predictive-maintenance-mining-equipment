# AI-Based Predictive Maintenance for Mining Equipment

Saskatchewan Polytechnic · Group 9 · Ganesh Mishra, Lakshya Sharma and Sagar Patel

A local application for classifying current component faults from supported sensor recordings and images. Eight mining-equipment mappings are supported: Pump, Haul Truck, Belt Conveyor, Dozer, Excavator, Wheel Loader, Drill Rig and Motor Grader. The datasets are laboratory component analogues, gear photographs and synthetic CAD renders, not mining-field measurements.

## Start here

- **Run predictions:** [Integrated prediction notebook](Notebook/Integrated_Equipment_Fault_Prediction.ipynb).
- **Understand the experiments:** [Condition-monitoring notebook](Notebook/Mining_Equipment_Condition_Monitoring.ipynb).
- **Application code:** [predictive_maintenance.py](Notebook/predictive_maintenance.py).
- **Run all automated checks:** [test_application.py](tests/test_application.py).

## Install and open in VS Code

Use Python 3.11 and open this repository folder in VS Code. Install the Python and Jupyter extensions.

```bash
git clone https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment.git
cd ai-based-predictive-maintenance-mining-equipment
python -m venv .venv
```

Activate the environment with `.venv\Scripts\activate` on Windows, or `source .venv/bin/activate` on Mac/Linux, then install:

```bash
python -m pip install -r requirements.txt
```

Open a notebook, select the `.venv` kernel and run its cells. The saved-result sections work without raw datasets. Real predictions require the local inputs, image indexes and trusted checkpoints described below. [requirements-lock.txt](requirements-lock.txt) records the previously tested package versions.

## Local data and trained models

Obtain source data from the [dataset manifest](Dataset_Manifest/manifest.md). Keep raw files in `Raw_Data/`, exported frames and their indexes in `Images/`, and deployment checkpoints in `Processed_Data/models/`. These folders are excluded from Git.

```bash
python Notebook/predictive_maintenance.py data-check
python Notebook/predictive_maintenance.py route-check
```

`route-check` performs real local inference: exit 0 means execution checks passed, 1 means a failure, and 2 means required local assets are missing. Execution checks are not independent accuracy estimates and may use fitting samples.

Every component checkpoint needs a verified SHA-256 entry in `Processed_Data/models/trusted_sha256.json`. Verify provenance and hashes before using `trust-models`; do not register an unknown model just to bypass a loading error. Training registers its own outputs. Joblib loading can execute code; image weights use trusted hashes and restricted weight loading.

The Motor Grader gear route uses `grader_gear_sensor.joblib` and `grader_gear_tooth.joblib`. Its original JUST slewing-bearing route remains separate. Restore verified checkpoints for demonstration rather than retraining automatically. Rebuild instructions, export requirements and previous verification records are preserved in the [experiment archive](Documentation/experiment_archive.md).

## Predict and demonstrate

```bash
python Notebook/predictive_maintenance.py serve
python Notebook/predictive_maintenance.py predict path/to/input.csv
python Notebook/predictive_maintenance.py demo-check
```

The upload page opens at `http://localhost:8613`, with a dashboard at `/dashboard`. It returns the equipment mapping, fault class, model probability and model name. Stop the local server with Ctrl+C. See the [demo script](Documentation/demo_script.md) for supported sample inputs and expected behavior.

Automatic identification recognizes supported schemas or verified image provenance. Shared CORA files and unresolved event-camera inputs remain ambiguous; select a compatible mapping only when its source is known. There is no validated universal whole-machine photograph classifier. Probabilities are uncalibrated model outputs, not failure risks or future failure dates.

## Results and scientific scope

The [evidence matrix](Documentation/equipment_evidence_matrix.md) records components, fault labels, evaluation groups and metrics. [Saved results](Results/README.md) and their [figures](Figures/) retain primary evaluations, time-split checks and unsuccessful follow-ups.

- Pump has the strongest aligned benchmark: 95.83% sensor and 100% visual balanced accuracy under different recording holdouts.
- CORA mappings have weak unseen-speed transfer. Dozer reuses Belt Conveyor data; Excavator and Wheel Loader also share some runs. They are not independent mining-machine trials.
- Drill Rig images are synthetic CAD renders. Sensor root cracks and rendered root breakage require a qualified comparison.
- Motor Grader's aligned gear comparison uses Aalto sensor balanced accuracy of 77.78% and tooth-image balanced accuracy of 82.59%, from different physical gears and evaluation units. The archived JUST bearing score is a separate component task.
- Time-split results measure later segments of recordings seen during training. They cannot replace unseen-speed or unseen-gear-pair results. No paired multimodal fusion, mining-field validation or remaining-life prediction is claimed.

## Essential documentation

| Document | Purpose |
|---|---|
| [Project report](Documentation/final_report.md) | Project summary, outcomes and limitations |
| [Methodology](Documentation/methodology.md) | Preprocessing, grouped evaluation and model selection |
| [Evidence matrix](Documentation/equipment_evidence_matrix.md) | Eight equipment mappings and comparable results |
| [Scientific validation](Documentation/scientific_validation.md) | Evidence checks and deployment qualifications |
| [Demo script](Documentation/demo_script.md) | Local demonstration workflow |
| [Experiment archive](Documentation/experiment_archive.md) | Consolidated original plans, investigations and verification history |

Final submission documents, videos and weekly exports are kept locally rather than in this repository. A fresh clone does not include the large datasets or trained model files.

## Automated checks

```bash
python -m pip install pytest pyflakes
python -m pyflakes Notebook tests
python -m pytest tests -q
```

All checks are organized by feature in one file, `tests/test_application.py`. Two real-file checks explicitly skip when their local data/checkpoints are unavailable. GitHub Actions runs the tests and lint checks on every push and pull request.

Code uses the [MIT License](LICENSE). Dataset publishers retain their licences and attribution, recorded in the manifest. No source datasets are redistributed here.
