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

## Recommended first subset

The full data has more than 38 million rows, so first select a bounded contiguous time range that:

1. fits in local memory after loading or is processed lazily;
2. contains maintenance events rather than using an unstratified row sample;
3. preserves time order and the context before each event; and
4. is documented by start/end timestamps, row count, source version, and sampling logic.

Do not randomly sample individual rows before building lag or rolling-window features. That would break the time-series context.

## Reproducibility record

For every shared result, note:

- upstream URL and commit/download date;
- source filename(s) and file checksum when practical;
- selected time range and all row filters;
- resampling frequency and aggregation rules;
- missing-value treatment;
- target horizon and maintenance exclusion/recovery window; and
- resulting row and event counts for each chronological split.
