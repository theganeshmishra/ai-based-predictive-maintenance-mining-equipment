# AI-Based Predictive Maintenance for Mining Equipment

Saskatchewan Polytechnic — PROJ 613 Capstone Project

**Team:** Ganesh Mishra, Lakshya Sharma, and Sagar Patel  
**Duration:** 12 weeks

## Project overview

This capstone has two independent tracks: predicting upcoming maintenance from mining-machine sensor data, and assessing visible mining-equipment damage with OpenCV. The sensor track uses the **Time-series Industrial Anomaly (TiNA)** dataset published by the DaSCI research group from an ArcelorMittal mining machine. The image track will use a separately sourced, labelled image dataset; the two tracks do not describe the same machine and their records will not be linked.

The official dataset describes more than 38 million observations and 108 variables over about 468 days, from September 2017 to December 2018. In our pilot file, those are 105 anonymized `FEATURE` columns plus `m_id`, `m_subid`, and `alarms`, alongside a `unix` timestamp (109 columns total). The source describes a two-second overall time step; the selected three-day pilot has a verified one-second cadence.

Official source: [ari-dasci/OD-TINA](https://github.com/ari-dasci/OD-TINA)

## Capstone objective

Develop and evaluate a reproducible proof of concept that predicts whether a maintenance event will occur within a defined future time window. Separately, use OpenCV to investigate visible damage on specific mining-equipment components. A still image can support current-damage detection; predicting future damage requires dated follow-up images or repair outcomes from the same equipment. Neither track should claim more than its labels support.

### Research questions

1. Which sensor, alarm, and rolling-window signals change before a maintenance event?
2. How accurately can baseline models identify a future maintenance window?
3. How do prediction horizon, class imbalance, and decision threshold affect operational usefulness?
4. Which features make the model's warnings interpretable to a maintenance team?

## Current work to review

For the Week 3 review, start with the project's **single TiNA notebook**,
[TiNA Predictive Maintenance notebook](notebooks/TiNA_Predictive_Maintenance.ipynb), and the
[short findings report](reports/week-03-data-quality-and-events.md).
Future weeks will extend this notebook and keep reusable code in `src/`.
The separate [OpenCV damage notebook](notebooks/OpenCV_Mining_Equipment_Damage.ipynb)
is an image-intake and preprocessing starter. It has no training data or
performance result yet; it does not add a second TiNA notebook.
The scripts and data notes support reproducibility. The submitted
[proposal](reports/Group_9_Capstone_Project_Proposal.pdf)
and [proposal slides](presentation/Group_9_Capstone_Project_Proposal_Presentation.pptx)
remain available without separate folder-index files.

## Data policy

**Do not commit TiNA source data, extracts, or trained models to Git.** The full dataset is too large for normal GitHub storage and has its own CC BY-SA 4.0 license. Small reviewed figures and metrics are allowed; raw data is not.

Each team member should download TiNA from the [official repository](https://github.com/ari-dasci/OD-TINA) and keep it in a local `data/raw/` directory. The repository's `.gitignore` excludes `data/raw/`, `data/interim/`, `data/processed/`, common tabular data formats, and model artifacts. See [data/README.md](data/README.md) for setup and sampling guidance.

The analysis should begin with a bounded, event-preserving time period. Record the exact date range, sampling rule, source commit or download date, and row count so that another team member can reproduce the subset.

## Getting started

Python 3.11 is recommended.

`requirements.txt` contains the dependencies needed for the Week 3 TiNA
notebook and the separate OpenCV image-intake starter; modeling packages will
be added when valid targets and evaluation data are ready.

On macOS or Linux:

```bash
git clone https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment.git
cd ai-based-predictive-maintenance-mining-equipment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/build_week2_subset.py
jupyter lab
```

On Windows PowerShell, use the same repository and a local environment:

```powershell
git clone https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment.git
cd ai-based-predictive-maintenance-mining-equipment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src\build_week2_subset.py
jupyter lab
```

If PowerShell blocks environment activation, run the same Python commands
through `.\.venv\Scripts\python.exe` and launch
`.\.venv\Scripts\jupyter.exe lab`; no execution-policy change is needed.
The notebook uses the pinned **September 8–10 three-day pilot**, not an
arbitrary local data file.

Before committing the TiNA notebook, restart its kernel and run all cells from top to bottom.

For a one-command local setup on macOS or Linux, run:

```bash
chmod +x run_local.sh
./run_local.sh
```

The launcher creates `.venv` when needed, installs the requirements, builds the
pinned three-day pilot if it is missing, and starts JupyterLab locally.

The [Week 2 acquisition record](reports/week-02-data-acquisition.md)
and [data dictionary](reports/data_dictionary.md) document the exact source,
checksum, selection rationale, and observed schema.

To reproduce the Week 3 findings, open
[`notebooks/TiNA_Predictive_Maintenance.ipynb`](notebooks/TiNA_Predictive_Maintenance.ipynb)
in VS Code, select the project's `.venv` Python kernel, and run all cells.

The [Week 3 data-quality and event-level report](reports/week-03-data-quality-and-events.md)
documents six maintenance episodes, provisional recovery gaps, model-readiness
limits, and the mining-equipment data scope. TiNA contains sensor and event
records, not images. The OpenCV notebook is independent: it looks for local
images under `data/raw/opencv/` and an optional local label file at
`data/raw/opencv/image_labels.csv`. These are ignored by Git. The notebook
can audit and preview images but makes no damage or future-failure prediction
until a suitable, labelled dataset is obtained.

## Separate OpenCV track

The team wants to assess more than one kind of mining equipment. Each image
must therefore identify its equipment type and component, and a model must be
evaluated separately for each equipment type before any broad claim is made.
The image source, usage permission, damage labels, and capture conditions must
be recorded. Candidate datasets are not treated as mining-machine evidence
merely because their components could also appear in a mine.

Use the [OpenCV notebook](notebooks/OpenCV_Mining_Equipment_Damage.ipynb) in VS
Code with the same `.venv` kernel as the TiNA notebook. It checks local image
files and optional labels, then shows preprocessing examples. For genuine
future-damage prediction, collect repeated dated inspections of the same
machine plus later damage or repair outcomes. Until then, report this track
as **visual damage detection/condition assessment**, not time-to-failure
prediction. Do not combine its records or metrics with TiNA.

## Analytical approach

- Load the large dataset lazily and inspect a bounded period before any full scan.
- Establish the timestamp, schema, missingness, event distribution, and maintenance recovery periods.
- Define a future prediction horizon and generate the target using only future event labels.
- Construct all predictor features from information available at prediction time.
- Split chronologically; never use a random train/test split for final evaluation.
- Begin with dummy, Naive Bayes, and interpretable linear baselines before tree-based models, once the target and chronological evaluation are viable.
- Evaluate precision, recall, F1, PR-AUC, false alarms, and event-level warning coverage. Accuracy alone is not appropriate for this imbalanced problem.
- Save small, shareable metrics and figures in `results/`; document large or excluded artifacts.

## 12-week delivery plan

| Week | Focus | Expected output |
|---|---|---|
| 1 | Scope, roles, repository, success criteria | Approved project charter and collaboration workflow |
| 2 | Data acquisition and data dictionary | Reproducible local data setup and bounded subset plan |
| 3 | Data understanding | Completed TiNA notebook's Week 3 analysis and data-quality log |
| 4 | Cleaning and preprocessing | Initial preprocessing code and a section in the single TiNA notebook |
| 5 | Exploratory analysis | Time-series, missingness, alarm, and maintenance findings |
| 6 | Target and feature design | Leakage-safe target definition and feature catalogue |
| 7 | Feature engineering | Reproducible features and a modelling-table summary in the TiNA notebook |
| 8 | Baseline modelling | Dummy, linear, and tree-based baseline results in the TiNA notebook |
| 9 | Validation and interpretation | Threshold analysis, error analysis, and feature importance |
| 10 | Final results and report | Reproducible results, limitations, and report draft |
| 11 | Presentation and review | Slide deck, demo flow, peer review, and revisions |
| 12 | Final submission | Final report, presentation, repository audit, and handoff |

## Team collaboration

Use GitHub Issues for every meaningful unit of work and assign one owner. Keep `main` stable; create short-lived branches such as `feature/eda-missingness` or `docs/report-methodology`. Submit a pull request, link the issue, and request review from a teammate before merging.

Avoid simultaneous edits to the same notebook. The issue owner should announce notebook ownership, clear large outputs before committing, and release ownership after merge. Prefer reusable logic in `src/` so it can be reviewed through readable text diffs.

Full conventions, review expectations, and a definition of done are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Reproducibility and ethics

- Fix random seeds and record the Python/package environment used for final results.
- Record all filtering, resampling, interpolation, and label-generation decisions.
- Treat the proof of concept as decision support, not an autonomous maintenance system.
- Discuss false negatives, false positives, sensor drift, anonymization, and transferability limits.
- Do not claim deployment readiness without prospective validation on the intended equipment and operating context.

## Dataset citation

```bibtex
@misc{tina_dasci_arcelor,
  title   = {Time-series Industrial Anomaly dataset},
  author  = {Ignacio Aguilera-Martos and David López and Marta García-Barzana and Julián Luengo and Francisco Herrera},
  year    = {2022},
  url     = {https://github.com/ari-dasci/OD-TINA}
}
```

## License

The original TiNA dataset is not redistributed here and remains subject to its source license, [CC BY-SA 4.0](https://github.com/ari-dasci/OD-TINA/blob/main/LICENSE). Original code and documentation in this repository are released under the [MIT License](LICENSE).
