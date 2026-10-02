# Week 3 addendum: master catalogue and 8-class prototype demo

**Status:** Prototype demo built and evaluated. The 8-class sensor model does **not** generalise to unseen days, and this addendum reports that result as found.

**Analysis date:** 2026-10-02. Code: [`prototype/`](../prototype/README.md). Seed 613.

## What was added

1. **Master catalogue** (`data/processed/master_catalog.csv`, local only): one row per sample from every dataset, with the same columns everywhere (dataset, sample id, equipment, modality, source file, label, label source, split group, time, licence or permission). It has 33,859 rows: 31,506 TiNA minutes, 512 Nonastreda samples and 1,841 bearing photos.
   The datasets are listed together but **not merged into one training table**. They come from different equipment and share no ids, columns or labels, so a merged model would learn which dataset a row came from rather than equipment health. This follows proposal slide 10.
2. **8-class TiNA machine-state model**, evaluated with whole days held out.
3. **Streamlit demo** (`streamlit run prototype/app.py`) that replays recorded data as a virtual sensor. No physical sensors are required.

## 8-class definition

No 8-class label exists in TiNA. The team defined 8 states per minute from the dataset's own `m_id`, `m_subid` and `alarms` columns, using the full September 2017 extract (6 to 30 Sep, 25 days, 1,889,531 one-second rows, 52 maintenance episodes).

| State | Minutes | Episodes |
|---|---:|---:|
| normal | 24,262 | |
| alarm (no maintenance) | 3,816 | |
| M4/MS3 | 2,317 | 16 |
| M4/MS5 | 526 | 9 |
| M9/MS1 | 245 | 10 |
| M9/MS6 | 114 | 9 |
| M2 | 134 | 3 |
| other_maintenance (M4/none, M7, M11, M13) | 92 | 5 |

Inputs are the per-minute mean and standard deviation of the 103 numeric sensors (206 features). Label columns are never used as inputs. Minutes with fewer than 30 seconds of data are dropped.

## Method

- Random forest, 300 trees, balanced class weights, fixed settings (no tuning).
- **Five blocks of five consecutive days**; each block held out once.
- **Chronological hold-out:** train on 6 to 22 Sep, test on 23 to 30 Sep. The demo replays these unseen days.
- For comparison only: a random-minute split, which is leaky because neighbouring minutes fall on both sides.

## Results

| Evaluation | Balanced accuracy | Macro-F1 | Accuracy | Maintenance episodes with correct type |
|---|---:|---:|---:|---:|
| Majority class ("normal") | 0.125 to 0.20 per block | | | 0 |
| 5-block day CV | 0.126 | 0.111 | 0.771 | 0 / 52 |
| Hold-out 23 to 30 Sep | 0.167 | 0.149 | 0.805 | 0 / 10 |
| *Leaky random-minute CV* | *0.946* | *0.935* | *0.975* | *n/a* |

- **On unseen days the model is no better than always predicting "normal".** Every block-CV score equals or barely exceeds the majority-class score (`results/prototype_8state_cv_by_block.csv`).
- The **77 to 80% accuracy is misleading**: "normal" alone is 77% of minutes. This repeats the Week 3 finding that accuracy hides missed events.
- The **leaky split gives 0.95**, about 7.5 times the honest score. The features identify *which day and time* a minute comes from, not the machine state.
- Two more checks, both still at chance and not used in the demo:
  - a simpler maintenance-vs-not version (balanced accuracy 0.50)
  - drift-corrected features, comparing each minute with the previous two hours (0.52)
- The hold-out days contain only 6 of the 8 states. M2 (3 episodes) and other_maintenance (5) are too rare to judge reliably whatever the model.

## Inspection module (separate data)

The demo's second step follows the proposal's alert → inspect design. It uses **Nonastreda** (Truchan and Ahmadi, 2025, DOI 10.17632/m892d2wtzh.1, CC BY 4.0): 512 milling-tool samples with force signals and microscope photos, in 3 wear classes. This is milling-tool wear, not mining equipment.

Under leave-one-tool-out (10 folds), signals and photos together reach a macro-F1 of 0.84 ± 0.09. The breakdown:
- photos alone: 0.85
- force signals alone: 0.59, below a run-count reference of 0.68
- label-shuffle check: 0.30, which is chance

The full analysis is in the sibling project `multimodal-pdm-from-scratch` (notebooks 01 to 03).

## Photo check (upload or camera)

The demo's first tab lets a user upload photos or take one with the camera. Each photo goes through:
1. OpenCV preprocessing, shown step by step: CLAHE, edges, texture heat map, and a damage-cue overlay for bearings
2. a model or screening rule
3. the five most similar reference photos
4. a familiarity check that flags photos unlike the training data (random noise and blank images were flagged in testing)

Results can be downloaded as a CSV report.

Wear models for each photo type, under leave-one-tool-out:
- machined surface: macro-F1 0.83 ± 0.07
- chips: 0.81 ± 0.05
- tool edge: 0.45 ± 0.14

Bearing photos get a rule-based screening, not a trained classifier, because no verified labels exist.

## Decision and next steps

- Present the 8-class TiNA result **as a negative finding**, showing why splitting by day matters. It is not a working classifier.
- Before more TiNA modelling:
  - get physical meanings, or at least sensor groupings, for the anonymised features
  - test longer windows and event-history features (proposal slide 11)
  - test per-day normalisation learned only on training days
  - consider fewer, better-supported classes (for example maintenance vs not, or M4 vs M9)
- Confirm the 8-class definition with the professor.
- Bearing photos still need human-verified damage labels and confirmed permission before any model result can be claimed.
