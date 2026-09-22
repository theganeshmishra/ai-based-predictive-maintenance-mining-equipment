# Week 2 — Data acquisition and subset plan

**Status:** Complete

**Completed:** 2026-09-21

**Team:** Ganesh Mishra, Lakshya Sharma, and Sagar Patel

## Source and reproducibility

- Official source: [ari-dasci/OD-TINA](https://github.com/ari-dasci/OD-TINA)
- Pinned source commit: `927b1584ea4aeac2ac7daa084a82f899b114419b`
- Download date: 2026-09-21
- Local combined file: `data/raw/tina_sample_2017-09-08_to_10.parquet`
- SHA-256: `19365a40e99083c37c9909f25b281caad6e47430c52b255c4c24b0db874db947`
- Local file size: approximately 2.5 MB Parquet
- In-memory size during verification: approximately 220.7 MB

The dataset file is intentionally excluded from Git. Each team member can
rebuild it from the pinned source by running:

```bash
python src/build_week2_subset.py
export TINA_DATA_PATH="data/raw/tina_sample_2017-09-08_to_10.parquet"
```

On Windows PowerShell, set the data path with:

```powershell
$env:TINA_DATA_PATH="data/raw/tina_sample_2017-09-08_to_10.parquet"
```

## Selected bounded period

The pilot covers the contiguous interval from **2017-09-08 00:00:00 UTC**
through **2017-09-10 23:59:59 UTC**.

| Check | Verified result |
|---|---:|
| Rows | 259,200 |
| Columns | 109 |
| Sampling interval | 1 second |
| Unique timestamps | 259,200 |
| Maintenance rows | 4,994 |
| Alarm rows | 26,719 |
| Null / NaN / infinite values | 0 / 0 / 0 |

This period was selected because it is small enough for laptop analysis while
preserving uninterrupted time order and including two maintenance classes,
three maintenance subtypes, and fourteen alarm codes. It is suitable for
workflow development and early analysis, but it is not representative enough
for final model conclusions.

## Decisions and limitations

- `unix` is the verified time field and must be sorted before time-based work.
- Random row sampling is prohibited before lag, rolling-window, or target creation.
- `m_id`, `m_subid`, and future alarm/maintenance information must not leak into predictors.
- Current maintenance rows will be excluded from prediction rows.
- A provisional 30-minute post-maintenance recovery exclusion will be tested in
  Week 3 and finalized after event-duration and gap analysis.
- Initial candidate prediction horizons are 30 minutes, 1 hour, and 4 hours;
  the final horizon belongs to the Week 6 target-design decision.
- Findings from this three-day period must be rechecked on broader chronological
  splits before they appear as final capstone claims.

## Week 3 handoff

Complete Notebook 01 using this selected period, convert consecutive maintenance
rows into event episodes, document recovery gaps, and add the data-quality log.
The detailed schema and pilot observations are in
[`data_dictionary.md`](data_dictionary.md).
