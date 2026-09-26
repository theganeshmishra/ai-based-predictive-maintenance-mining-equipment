# AI-Based Predictive Maintenance for Mining Equipment

Saskatchewan Polytechnic — PROJ 613 Capstone Project

**Team:** Ganesh Mishra, Lakshya Sharma, and Sagar Patel  
**Duration:** 12 weeks

## Project overview

This capstone investigates whether anonymized industrial sensor and alarm data can be used to identify operating patterns that precede maintenance events on mining equipment. The work uses the **Time-series Industrial Anomaly (TiNA)** dataset published by the DaSCI research group from sensor data collected on an ArcelorMittal mining machine.

The official dataset contains more than 38 million observations and 108 features at an approximately two-second sampling interval. It covers about 468 days from September 2017 to December 2018. Maintenance events are represented by `m_id` and `m_subid`; alarm events are represented by `alarms`.

Official source: [ari-dasci/OD-TINA](https://github.com/ari-dasci/OD-TINA)

## Capstone objective

Develop and evaluate a reproducible proof of concept that predicts whether a maintenance event will occur within a defined future time window. The final model should support early-warning analysis while avoiding time-series leakage and clearly documenting practical limitations.

### Research questions

1. Which sensor, alarm, and rolling-window signals change before a maintenance event?
2. How accurately can baseline models identify a future maintenance window?
3. How do prediction horizon, class imbalance, and decision threshold affect operational usefulness?
4. Which features make the model's warnings interpretable to a maintenance team?

## Repository structure

```text
.
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_TiNA_Data_Understanding.ipynb
│   ├── 02_TiNA_Preprocessing_EDA.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   └── 04_Baseline_Models.ipynb
├── src/
│   ├── __init__.py
│   ├── build_week2_subset.py
│   └── data_utils.py
├── reports/
│   ├── README.md
│   ├── data_dictionary.md
│   ├── week-02-data-acquisition.md
│   └── week-03-data-quality-and-events.md
├── presentation/
│   └── README.md
├── results/
│   └── README.md
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── requirements.txt
```

## Data policy

**Do not commit the full TiNA dataset, extracts, trained models, or generated artifacts to Git.** The source data is much too large for normal GitHub storage and has its own CC BY-SA 4.0 license.

Each team member should download TiNA from the [official repository](https://github.com/ari-dasci/OD-TINA) and keep it in a local `data/raw/` directory. The repository's `.gitignore` excludes `data/raw/`, `data/interim/`, `data/processed/`, common tabular data formats, and model artifacts. See [data/README.md](data/README.md) for setup and sampling guidance.

The analysis should begin with a bounded, event-preserving time period. Record the exact date range, sampling rule, source commit or download date, and row count so that another team member can reproduce the subset.

## Getting started

Python 3.11 is recommended.

```bash
git clone https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment.git
cd ai-based-predictive-maintenance-mining-equipment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Place the downloaded data under `data/raw/`. The local starter sample uses the official September 6, 2017 Parquet partition and the source timestamp field `unix`:

```bash
export TINA_DATA_PATH="data/raw/tina_sample_2017-09-06.parquet"
export TINA_TIME_COLUMN="unix"
```

The one-day file is only an environment starter. Notebook 01 now uses the
documented **September 8–10 three-day pilot** and requires that file locally.

The notebooks are intended to run in numerical order. Before committing a notebook, restart its kernel and run all cells from top to bottom.

For a one-command local setup on macOS or Linux, run:

```bash
chmod +x run_local.sh
./run_local.sh
```

The launcher creates `.venv` when needed, installs the requirements, builds the
pinned three-day pilot if it is missing, and starts JupyterLab locally. The
one-day starter remains optional for a quick environment check, but Notebook 01
uses the three-day pilot.

For the verified Week 2 event-preserving pilot, run
`python src/build_week2_subset.py` and point `TINA_DATA_PATH` to the generated
three-day Parquet file. The [Week 2 acquisition record](reports/week-02-data-acquisition.md)
and [data dictionary](reports/data_dictionary.md) document the exact source,
checksum, selection rationale, and observed schema.

To reproduce the Week 3 findings, open
[`notebooks/01_TiNA_Data_Understanding.ipynb`](notebooks/01_TiNA_Data_Understanding.ipynb)
in VS Code, select the project's `.venv` Python kernel, and run all cells.

The [Week 3 data-quality and event-level report](reports/week-03-data-quality-and-events.md)
documents six maintenance episodes, provisional recovery gaps, model-readiness
limits, and the selected candidate image dataset for the separate CV module.

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
| 3 | Data understanding | Completed notebook 01 and data-quality log |
| 4 | Cleaning and preprocessing | Initial preprocessing pipeline in notebook 02 |
| 5 | Exploratory analysis | Time-series, missingness, alarm, and maintenance findings |
| 6 | Target and feature design | Leakage-safe target definition and feature catalogue |
| 7 | Feature engineering | Completed notebook 03 and modelling table |
| 8 | Baseline modelling | Dummy, linear, and tree-based baselines in notebook 04 |
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
