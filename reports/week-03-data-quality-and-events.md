# Week 3 — TiNA data quality, events, and exploratory baseline

**Status:** Week 3 pilot analysis and a six-day exploratory baseline completed. Final model validation remains open.

**Analysis dates:** Three-day pilot analyzed 2026-09-25 and revalidated 2026-09-26; six-day baseline analyzed 2026-10-01 (America/Regina).

**Revalidation environment:** Python 3.9.6, Polars 1.36.1, pandas 2.3.3,
Matplotlib 3.9.4, PyArrow 21.0.0, and scikit-learn 1.6.1 for the baseline.
See `requirements.txt` for supported version ranges.

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

This explains why the original three-day pilot cannot support a model score.
The six-day exploratory baseline below is a separate extension, not a score
from the three-day split.

## Six-day exploratory baseline

To address the professor's **Naive Bayes, feature selection, and accuracy**
feedback, the team extended the bounded source period to **2017-09-07
00:00:00 through 2017-09-12 23:59:59 UTC**. It uses six complete one-day
Parquet partitions from the same [pinned official TiNA commit](https://github.com/ari-dasci/OD-TINA/tree/927b1584ea4aeac2ac7daa084a82f899b114419b).
`src/build_week2_subset.py` verifies each partition's SHA-256 before use;
`python -m src.week3_baseline` reproduces the modeling results. Source data
remain under ignored `data/raw/`, not on GitHub. The three newly required
partition hashes are:

| Date | SHA-256 |
|---|---|
| 2017-09-07 | `031380fd9c6655579b53bcafb7e485a681ea5d978ab8beeba2414b92090a1fb0` |
| 2017-09-11 | `713db757201caff72efb91b0b123697304a188380fe7c6051a6d2f4954d728b1` |
| 2017-09-12 | `4ef82756e263b4198d4e635d348d4482e80cde443878dd3c1824138de086487b` |

The extension contains **518,400 rows, 109 columns, 103 numeric and two
string sensor features, 8,135 maintenance-coded rows, 34,070 alarm-coded
rows, and nine distinct maintenance episodes**. Timestamps are unique and
exactly one second apart after sorting. We found no null, NaN, or infinite
values and no sensor constant across the full six days. The six-day findings
do **not** erase the eight features that were constant in the earlier
three-day pilot; constancy depends on the period inspected.

### Exploratory method

- Aggregate the numeric sensors to one row per complete minute using that
  minute's mean and standard deviation (206 candidate predictors). A warning
  made at second 59 uses no later sensor data. Exclude `m_id`, `m_subid`, and
  `alarms` from predictors. String sensors are not encoded in this baseline.
- Set the target to **a new maintenance episode beginning within the next
  30 minutes**, not current maintenance. Exclude active-maintenance minutes
  and the following 30 minutes as a provisional recovery rule. Purge the last
  30 minutes of each chronological split so labels never cross split bounds.
- Train on September 7–10 (**six onsets**), validate on September 11 (**one**),
  and evaluate on September 12 (**two**). This split is fixed by date, not a
  shuffled row split. These dates were chosen after inspecting the event
  distribution, so the test day is not an independent prospective sample.
  Eligible training minutes: **5,467**, with **180**
  positive target minutes.
- Fit median imputation, zero-variance filtering, scaling, and a top-20
  univariate ANOVA feature selector using **training data only**. Compare
  Gaussian Naive Bayes against a most-frequent dummy baseline at a fixed 0.5
  score threshold; no test-set threshold tuning or calibration. The selected
  predictors are `FEATURE28_mean`, `FEATURE29_mean`, `FEATURE30_mean`,
  `FEATURE33_mean`, `FEATURE35_mean`, `FEATURE36_mean`, `FEATURE72_mean`,
  `FEATURE78_mean`, `FEATURE79_mean`, `FEATURE80_mean`, `FEATURE81_mean`,
  `FEATURE83_mean`, `FEATURE90_mean`, `FEATURE91_mean`, `FEATURE95_mean`,
  `FEATURE97_mean`, `FEATURE98_mean`, `FEATURE99_mean`, `FEATURE100_mean`,
  and `FEATURE103_mean`. Their anonymized names do not identify components.

| Split / model | Eligible min (positive) | Accuracy | Balanced acc. | Precision | Recall | F1 | PR-AUC | Episodes warned | False-alarm min / runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation / dummy | 1,342 (30) | 0.9776 | 0.5000 | 0 | 0 | 0 | 0.0224 | 0/1 | 0 / 0 |
| Validation / Gaussian NB | 1,342 (30) | 0.0484 | 0.5133 | 0.0230 | 1.0000 | 0.0449 | 0.0230 | 1/1 | 1,277 / 4 |
| Test / dummy | 1,335 (60) | 0.9551 | 0.5000 | 0 | 0 | 0 | 0.0449 | 0/2 | 0 / 0 |
| Test / Gaussian NB | 1,335 (60) | 0.5311 | 0.7545 | 0.0875 | 1.0000 | 0.1609 | 0.0875 | 2/2 | 626 / 8 |

The dummy's **95.5% test accuracy misses both events**. Gaussian Naive Bayes
flags both, but **626 false-alarm minutes** and only **8.8% precision** make
it operationally unsuitable at this threshold. Episode coverage is only
`2/2`, not evidence of general sensitivity. PR-AUC is average precision over
minute-level scores; overlapping 30-minute windows are highly dependent, so
minute counts must not be interpreted as independent trials. False-alarm runs
are consecutive false-positive eligible minutes, separated when the gap is
over one minute.

**Decision:** This completes the requested Week 3 *exploratory* model and
feature-selection work, but not a validated maintenance predictor. Nine total
episodes, one validation event, two test events, date selection after inspecting
events, and a short single-machine period cannot establish stable performance.
Before final modeling, expand
chronological coverage substantially, revisit the 30-minute horizon and
recovery rule, assess drift and feature stability, and reduce false alarms.
Do not tune repeatedly on the September 12 test day or infer a repair type
from anonymized sensors.

## Mining-equipment scope clarification

The capstone's current evidence is TiNA sensor, alarm, and maintenance data
from mining equipment. TiNA does not provide equipment images or video, so
OpenCV image analysis cannot be run on this dataset. A separate visual module
should only be considered if the team obtains a relevant mining-equipment
image source and defines labels and permissions. No computer-vision model or
performance result is claimed for Week 3.

## Addendum

The master catalogue, the 8-class TiNA machine-state evaluation and the prototype demo are in
[`week-03-prototype-addendum.md`](week-03-prototype-addendum.md).
