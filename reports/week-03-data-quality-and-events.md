# Week 3 — TiNA data quality and event-level EDA

**Status:** Completed for the three-day pilot; broader-period validation remains open.

**Analysis date:** 2026-09-25; notebook revalidated 2026-09-26

**Revalidation environment:** Python 3.9.6, Polars 1.36.1, pandas 2.3.3,
Matplotlib 3.9.4, and PyArrow 21.0.0. See `requirements.txt` for supported
version ranges.

**Input:** `data/raw/tina_sample_2017-09-08_to_10.parquet` from the pinned
[official TiNA source](https://github.com/ari-dasci/OD-TINA) documented in the
[Week 2 acquisition record](week-02-data-acquisition.md). The Parquet file is
local only and is not committed to Git.

Reproduce the checks by running the single TiNA notebook,
[`TiNA_Predictive_Maintenance.ipynb`](../notebooks/TiNA_Predictive_Maintenance.ipynb),
from top to bottom after building the pilot with `python src/build_week2_subset.py`.

## Data-quality log

| Check | Pilot finding | Decision |
|---|---:|---|
| Rows / columns | 259,200 / 109 | Retain the full contiguous three-day pilot. |
| UTC period | 2017-09-08 00:00:00 to 2017-09-10 23:59:59 | Use `unix` as the time index. |
| Unique timestamps / cadence | 259,200 / exactly 1 second | Preserve chronological order. |
| Null / NaN / infinite values | 0 / 0 / 0 | Do not invent an imputation need for this pilot; recheck broader data. |
| Maintenance/subtype missing-code mismatches | 0 | Keep both fields as labels/event metadata, not predictive inputs. |
| Numeric sensor features constant in this pilot | 8 | Reassess using training data from the eventual broader period. |
| String sensor features | 2 | Handle separately from numeric sensors during preprocessing. |
| Maintenance-coded / alarm-coded rows | 4,994 / 26,719 | Row counts are not independent event counts. |

The eight pilot-constant sensors are `FEATURE9`, `FEATURE10`, `FEATURE19`,
`FEATURE20`, `FEATURE41`, `FEATURE53`, `FEATURE62`, and `FEATURE63`. The two
string sensor fields are `FEATURE76` and `FEATURE87`. Physical meanings and
units for the anonymized features are unavailable; no component-level meaning
is inferred from their names.

The notebook also plots maintenance-coded and alarm-coded **rows per hour**.
This is a descriptive time-series view, not an independent-event count or a
predictive result.

## Maintenance episodes and gaps

An episode is a maximal sequence of consecutive one-second rows where
`m_id != 'none'`. A change of code within an uninterrupted active sequence
would remain in one episode and all codes would be retained. A gap is the
number of non-maintenance seconds between one episode's last row and the
next episode's first row. Alarm counts below are *rows*, not distinct alarm
events.

| Episode | Start (UTC) | End (UTC) | Duration (s) | Gap before (s) | Maintenance / subtype | Alarm rows in prior 30 min / during |
|---:|---|---|---:|---:|---|---:|
| 1 | 2017-09-09 00:33:36 | 2017-09-09 00:46:09 | 754 | — | M9 / MS6 | 0 / 0 |
| 2 | 2017-09-09 07:47:17 | 2017-09-09 07:53:27 | 371 | 25,267 | M9 / MS6 | 0 / 0 |
| 3 | 2017-09-09 16:20:19 | 2017-09-09 16:28:06 | 468 | 30,411 | M4 / MS5 | 0 / 0 |
| 4 | 2017-09-09 20:04:03 | 2017-09-09 20:17:10 | 788 | 12,956 | M4 / MS3 | 0 / 0 |
| 5 | 2017-09-10 03:57:00 | 2017-09-10 04:17:21 | 1,222 | 27,589 | M9 / MS6 | 0 / 0 |
| 6 | 2017-09-10 05:20:07 | 2017-09-10 05:43:17 | 1,391 | 3,765 | M9 / MS6 | 1,355 / 1,391 |

Thus **4,994 maintenance rows represent six episodes**, with durations of
371–1,391 seconds. The shortest gap is 3,765 seconds (1 hour, 2 minutes,
45 seconds). Only the sixth episode has alarm-coded rows in the immediately
preceding 30-minute window. Six events are too few to infer that alarms are
generally predictive or non-predictive.

## Provisional recovery exclusion

Excluding active maintenance rows and the next 1,800 seconds after each
episode would remove 4,994 + 10,800 rows, leaving **243,406 pilot rows** for
possible prediction times. None of the six 30-minute recovery windows overlap
a subsequent episode in this pilot because the shortest between-episode gap
exceeds 1,800 seconds. This is a *sensitivity check*, not a finalized target
or universal maintenance recovery rule. Test different windows on a broader
period before using one for modeling.

## Modeling readiness and professor feedback

A provisional chronological 70/15/15 **row** split starts validation at
2017-09-10 02:24:00 UTC and test at 2017-09-10 13:12:00 UTC. It puts
**4 / 2 / 0 distinct maintenance onsets** into training / validation / test.
The test segment has no maintenance episode; model accuracy there would not
measure the ability to warn before maintenance. A classifier predicting that
maintenance is never active would already be correct on 98.07% of *all pilot
rows*, while detecting no maintenance. This is only an imbalance illustration, not a future-warning
model result.

Before the requested **Naive Bayes, feature selection, and accuracy-based
comparison**:

1. Expand to a longer chronological period with maintenance episodes in each
   validation and test segment; record episode counts, not only row counts.
2. Finalize a future-event prediction horizon and the post-event exclusion
   rule. Exclude `m_id`, `m_subid`, and any future alarm or maintenance values
   from predictors.
3. Fit constant/low-information filtering and other feature selection on
   training data only. Reassess the eight pilot-constant sensors on the new
   training period. Choose the number of features on validation data, not test.
4. Compare Naive Bayes with a most-frequent dummy baseline and other planned
   models. Report accuracy **alongside** balanced accuracy, precision, recall,
   F1, PR-AUC, and event-level warning coverage; inspect missed events and
   false-alarm episodes. Report the untouched test result once.

No predictive model or performance score is claimed for Week 3.

## Computer-vision dataset decision

For Sagar's separate, preliminary visible-defect module, the selected
candidate is the [official Kolektor Surface-Defect Dataset 2 (KSDD2)](https://www2.vicos.si/resources/kolektorsdd2/).
The publisher describes 356 images with visible defects and 2,979 without,
including an existing train/test split and surface flaws such as scratches
and spots. Its license is **CC BY-NC-SA 4.0**. It is a controlled
industrial-production dataset, **not mining equipment**, so any CV result
would demonstrate a method, not validated transfer to mining machinery. Keep
the images local and preserve their separate license and attribution; do not
relicense them as MIT. Download and class-label checks are still pending; no
CV model has been trained.
