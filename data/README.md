# TiNA data setup

The data directory stores local inputs and derived datasets. **Do not commit the TiNA dataset or generated data files to GitHub.** Only this instruction file should be versioned.

## Source

Download or clone TiNA from the official repository:

- [ari-dasci/OD-TINA](https://github.com/ari-dasci/OD-TINA)
- Dataset license: CC BY-SA 4.0

Record the upstream commit or download date in the experiment notes. Do not copy or re-license the source data under this project's MIT License.

## Local layout

```text
data/
├── raw/          # untouched source files
├── interim/      # temporary cleaned or sampled files
└── processed/    # model-ready feature tables
```

These directories and common data formats are ignored by Git. Create the directories locally as needed.

## One-day local starter sample

The official dataset is partitioned by day. To run the notebooks without downloading the full dataset, start with the September 6, 2017 partition (about 74,000 rows):

```bash
mkdir -p data/raw
curl -fL \
  "https://raw.githubusercontent.com/ari-dasci/OD-TINA/main/data/day%3D2017-09-06/data.parquet" \
  -o data/raw/tina_sample_2017-09-06.parquet
export TINA_DATA_PATH="data/raw/tina_sample_2017-09-06.parquet"
export TINA_TIME_COLUMN="unix"
```

This sample is suitable for verifying the environment only. Notebook 01 now
uses the three-day pilot below; the one-day starter is not event-diverse enough
even for the pilot event-level analysis.

## Recommended first subset

The full data has more than 38 million rows, so first select a bounded contiguous time range that:

1. fits in local memory after loading or is processed lazily;
2. contains maintenance events rather than using an unstratified row sample;
3. preserves time order and the context before each event; and
4. is documented by start/end timestamps, row count, source version, and sampling logic.

Do not randomly sample individual rows before building lag or rolling-window features. That would break the time-series context.

## Verified Week 2 pilot subset

The project uses the contiguous period from September 8 through September 10,
2017 for its reproducible Week 2 pilot. It contains 259,200 rows, 4,994 rows
with a maintenance code, and 26,719 rows with an alarm code. Build the local
file directly from the pinned official source:

```bash
python src/build_week2_subset.py
export TINA_DATA_PATH="data/raw/tina_sample_2017-09-08_to_10.parquet"
```

On Windows PowerShell, set the data path with:

```powershell
$env:TINA_DATA_PATH="data/raw/tina_sample_2017-09-08_to_10.parquet"
```

Expected SHA-256:
`19365a40e99083c37c9909f25b281caad6e47430c52b255c4c24b0db874db947`.
See [`reports/week-02-data-acquisition.md`](../reports/week-02-data-acquisition.md)
for the selection rationale, verification results, and limitations.
Run [`notebooks/01_TiNA_Data_Understanding.ipynb`](../notebooks/01_TiNA_Data_Understanding.ipynb)
to reproduce the data-quality checks, maintenance episodes, recovery-gap
checks, and chronological split warning summarized in
[`reports/week-03-data-quality-and-events.md`](../reports/week-03-data-quality-and-events.md).

## Reproducibility record

For every shared result, note:

- upstream URL and commit/download date;
- source filename(s) and file checksum when practical;
- selected time range and all row filters;
- resampling frequency and aggregation rules;
- missing-value treatment;
- target horizon and maintenance exclusion/recovery window; and
- resulting row and event counts for each chronological split.
