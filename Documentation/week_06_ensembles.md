# Week 6 (proposal): advanced ensemble models — XGBoost / LightGBM comparison

**October 2026 — Saskatchewan Polytechnic PROJ 613.** The proposal's Week 6 deliverable is an
"XGBoost/LightGBM comparison table". This was added on top of the professor's Week 6 remarks
([Week 6: professor's remarks](week_06_professor_remarks.md)).

## What was done

- New command for every machine: `python Models/<machine>.py ensembles` (add `--split time` for the
  secondary time-split test). Each notebook has a matching section.
- Models compared on the **same CSV sensor features** and the **same held-out recordings** as the Week 6
  CSV main test: majority-class control, random forest, extra trees, scikit-learn gradient boosting,
  **XGBoost** (3.2.0) and **LightGBM** (4.7.0).
- Settings were fixed in advance (300 trees, learning rate 0.05, shallow trees); nothing was tuned on the
  test. One prediction per recording (mean class probabilities of its windows). Training time is recorded.
- Feature set: the base set (Pump, Haul Truck) or plain vibration set (Belt Conveyor, Excavator), the same
  one the saved `train-final` model uses. The Week 6 "validation-selected" model can choose richer feature
  sets, so its scores (Haul Truck 0.800, Belt Conveyor 0.625) are not directly comparable to this table.

## Results: accuracy on the main test (macro F1 in brackets)

| Model | Pump (24) | Haul Truck (30) | Belt Conveyor (16) | Excavator (16) |
|---|---:|---:|---:|---:|
| Majority class (control) | 0.250 | 0.500 | 0.250 | 0.250 |
| Random forest | **0.958** (0.958) | **0.867** (0.831) | 0.375 (0.288) | 0.250 (0.209) |
| Extra trees | **0.958** (0.958) | 0.833 (0.780) | 0.438 (0.356) | **0.312** (0.265) |
| Gradient boosting (scikit-learn) | **0.958** (0.958) | 0.800 (0.734) | 0.438 (0.393) | 0.250 (0.218) |
| **XGBoost** | **0.958** (0.958) | 0.833 (0.775) | 0.438 (0.372) | 0.250 (0.212) |
| **LightGBM** | 0.917 (0.916) | 0.800 (0.733) | **0.562** (0.484) | **0.312** (0.265) |

Numbers in brackets after the machine = test recordings. Time split (secondary): every model scores
1.000 on Haul Truck and Belt Conveyor and 0.938 on Excavator.

Training time for all test rounds together: 0.3–2.6 s for random forest, extra trees, XGBoost and
LightGBM; 1.7–8.1 s for scikit-learn gradient boosting (CPU).

## What it means

- **XGBoost and LightGBM do not beat the random forest** where the data allow a good model (Pump,
  Haul Truck). On the Pump every tree ensemble gets 23/24 recordings.
- On the CORA machines LightGBM is the best model (Belt Conveyor 9/16, Excavator 5/16), but all models stay
  far below 90% on the unseen-speed test, so the limit is the data (one run per fault per speed), not the
  model family.
- With 16–30 test recordings, one recording moves accuracy by 3–6 points, so differences of one or two
  recordings between models are not meaningful.
- Decision: keep the random forest as the main sensor model (simplest, fastest, never worse than the
  boosted models by more than one recording on Pump/Haul Truck).
