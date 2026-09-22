# TiNA data dictionary

This dictionary documents the schema used by the capstone. The source is the
official [ari-dasci/OD-TINA](https://github.com/ari-dasci/OD-TINA) repository,
pinned for the Week 2 pilot to commit
`927b1584ea4aeac2ac7daa084a82f899b114419b`.

## Schema overview

| Columns | Count | Type in pilot subset | Role | Interpretation |
|---|---:|---|---|---|
| `FEATURE0`–`FEATURE104` | 105 | 103 `Float64`, 2 `String` | Predictors | Anonymized equipment signals. Physical names and units are not published, so the project must not invent domain meanings. |
| `m_id` | 1 | `String` | Maintenance event | Primary maintenance code. `none` means no recorded maintenance on that row. |
| `m_subid` | 1 | `String` | Maintenance subtype | More specific maintenance code. `none` means no recorded subtype on that row. |
| `alarms` | 1 | `String` | Alarm event/predictor | Alarm code active on that row. `none` means no recorded alarm. |
| `unix` | 1 | `Int64` | Time/index | Unix timestamp in UTC seconds. The selected pilot has a regular one-second cadence. |

`FEATURE76` and `FEATURE87` are stored as strings in the selected subset. All
other anonymized feature columns are floating-point values. The event columns
must be excluded from predictors whenever they would reveal the current or
future target.

## Codes observed in the Week 2 pilot

The selected period is 2017-09-08 00:00:00 through 2017-09-10 23:59:59 UTC.

| Field | Observed values and row counts |
|---|---|
| `m_id` | `none` 254,206; `M4` 1,256; `M9` 3,738 |
| `m_subid` | `none` 254,206; `MS3` 788; `MS5` 468; `MS6` 3,738 |
| `alarms` | `none` 232,481; `A111` 5,535; `A115` 3,684; `A12` 3,613; `A97` 2,240; `A128` 1,814; `A35` 1,810; `A13` 1,808; `A34` 1,806; `A116` 1,802; `A118` 1,801; `A5` 794; `A95` 6; `A23` 4; `A96` 2 |

These are observations from the pilot window, not a claim that the full TiNA
dataset contains no other codes.

## Data-quality notes

- The pilot contains 259,200 rows and 109 columns.
- All 259,200 timestamps are unique, sorted, and separated by exactly one second.
- No null, NaN, or infinite values were observed in this period.
- Eight sensor columns are constant in this short window: `FEATURE9`,
  `FEATURE10`, `FEATURE19`, `FEATURE20`, `FEATURE41`, `FEATURE53`, `FEATURE62`,
  and `FEATURE63`. They may vary elsewhere and must be reassessed on expanded data.
- Maintenance appears on 4,994 rows (1.93%); alarms appear on 26,719 rows
  (10.31%). Row counts are not event counts because a single event can span
  consecutive rows.
- The raw feature meanings and units are anonymized. Analysis should describe
  statistical behaviour rather than assigning unsupported physical meanings.
