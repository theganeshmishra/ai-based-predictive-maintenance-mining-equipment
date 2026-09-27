# Week 2 — Data acquisition and subset plan

**Status:** Complete

**Completed:** 2026-09-21

**Team:** Ganesh Mishra, Lakshya Sharma, and Sagar Patel

## Source and reproducibility

- Official source: [ari-dasci/OD-TINA](https://github.com/ari-dasci/OD-TINA)
- Pinned source commit: `927b1584ea4aeac2ac7daa084a82f899b114419b`
- Download date: 2026-09-21
- Local combined file: `data/raw/tina_sample_2017-09-08_to_10.parquet`
- Original local output SHA-256: `19365a40e99083c37c9909f25b281caad6e47430c52b255c4c24b0db874db947` (reference only; combined Parquet bytes can vary by writer version)
- Local file size: approximately 2.5 MB Parquet
- In-memory size during verification: approximately 220.7 MB

The dataset file is intentionally excluded from Git. Each team member can
rebuild it from the pinned source by running:

```bash
python src/build_week2_subset.py
```

The Week 3 notebook uses this pinned path directly. On Windows PowerShell,
run `python src\build_week2_subset.py` after activating `.venv`.
The script verifies these official daily-partition SHA-256 checksums before
combining them:

| Date | Source partition SHA-256 |
|---|---|
| 2017-09-08 | `7389a5347a95b6a03018879380859b68ac811fec939be297732696dd46157977` |
| 2017-09-09 | `ca44e34ec06dcccb76be106d095b725a1926659911623bae9d0b907a1ddc3bb5` |
| 2017-09-10 | `a427206041323dfe4e5e61a0eb4a6ab204059540fbed056cffdfde4e3f196223` |

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

## Week 3 follow-up

The [TiNA notebook](../notebooks/TiNA_Predictive_Maintenance.ipynb) uses this
period to group maintenance rows into episodes, document recovery gaps, and
check data quality. Its findings and modeling limitations are in the
[Week 3 report](week-03-data-quality-and-events.md). The detailed schema and
pilot observations remain in [`data_dictionary.md`](data_dictionary.md).
