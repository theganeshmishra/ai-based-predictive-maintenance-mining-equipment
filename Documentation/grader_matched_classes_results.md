# Grader with matched classes: results

AI-Based Predictive Maintenance for Mining Equipment · Group 9 · November 7, 2026 · [plan](grader_matched_classes_plan.md) committed before running (c92b686)

## Why

The Grader sensor model (slewing bearing: healthy, inner ring, outer ring, rolling element) and the Grader images (gear teeth: good vs damaged) answer different questions. On November 7 we searched the web again for photo or thermal-image data with the four bearing classes and found none with a public licence. The search turned up papers with thermal images (for example [Mehta et al. 2021](https://onlinelibrary.wiley.com/doi/10.1155/2021/9947300)) but no released data, and [BD-DET](https://ieee-dataport.org/documents/bearing-defect-dataset), which covers seal-cover scratches rather than ring or rolling-element faults. So the sensor side was moved to the image side's task instead: **healthy gear vs damaged tooth**. The circle drive of a motor grader is turned by a pinion gear, so this is a relevant analogue.

The archived Grader results and the deployment checkpoints are unchanged. The new results are in `Results/experiments/grader_gear/` and the figures in `Figures/experiments/grader_gear/`. Rerun with `python Notebook/pdm.py grader-gear mcc5` and `grader-gear aalto`.

## Results (each run once)

| Side | Data | Test | Units | Balanced accuracy | Accuracy | Macro F1 | Majority control (balanced / accuracy) |
|---|---|---|---|---|---|---|---|
| Images (existing) | Roboflow GearBox_adddingImg v1, real photos | each video or photo set held out (55 sets) | 2,879 tooth crops | **82.6%** | 97.2% | 85.8% | n/a |
| Sensor | Aalto Gear Failure Dataset 3012 Hz, thruster rig | **unseen gear pair** (8 rounds) | 425 runs | **77.8%** | 78.3% | 76.6% | 50.0% / 66.6% |
| Sensor | MCC5-THU gearbox | unseen speed (3 rounds) | 60 recordings | **70.8%** | 88.3% | 76.0% | 50.0% / 80.0% |

Confusion matrices (rows actual, columns predicted; healthy, damaged):

- Aalto: `[[108, 34], [58, 225]]`. Healthy recall 76.1% (108 of 142), damaged recall 79.5% (225 of 283).
- MCC5-THU: `[[5, 7], [0, 48]]`. All 48 damaged recordings caught, but only 5 of 12 healthy ones.
- Images: `[[2690, 24], [56, 109]]`. Good teeth 99.1%, damaged teeth 66.1%.

## Details

**Aalto** (Dahl et al., [Mendeley fywnj597d8 v2](https://data.mendeley.com/datasets/fywnj597d8/2), CC BY 4.0, [data paper](https://doi.org/10.1016/j.dib.2024.111171)):
- 8 gear pairs (GP1–7, GP9). Each has about 18 healthy runs (6 drive speeds, 250–1500 rpm, × 3 loads) and one replicated fault installed twice.
- The faults are tooth flank fracture, pitting, micropitting and wear, each mild or severe. All are counted as damaged.
- 30 evenly spaced 1 s windows per run, after the first 10 s.
- Each round holds out one gear pair, with all of its healthy and faulty runs. The feature set and model are chosen inside the remaining 7 pairs.
- Chosen per round: acceleration + shaft orders + torque in 6 of 8 rounds (extra trees, SVM or logistic regression); acceleration + orders once; acceleration alone once.

**MCC5-THU:**
- Damaged = tooth break (light, medium, heavy) and missing tooth. 12 healthy and 48 damaged recordings.
- The same 1 s window and shaft-order features as improvement attempt A. Chosen per round: vibration (twice) and vibration + orders.

## What this shows, and what it does not

- Both sides address tooth damage at different scales. A sensor run describes the gear pair, while a photo crop describes only the visible tooth. The two sensor results come from different rigs, and both beat the majority control by a clear margin on held-out gears (Aalto) or speeds (MCC5-THU).
- **Aalto is the stronger sensor test.** It holds out whole physical gear pairs, has 425 runs, covers four fault types, and is independent of every other machine in the project. Its 77.8% balanced accuracy is close to the image side's 82.6%, but under 80%.
- **MCC5-THU** shares its rig and healthy recordings with the Drill Rig, so it is not independent. Healthy gears at an unseen speed are its weak spot, as in the Drill Rig result.
- On Aalto, each pair's healthy runs and fault runs are separate installations, and the data paper notes installation-dependent amplitude. The model may partly detect a re-installation rather than the damage. Holding out whole pairs prevents memorising a pair, but it does not remove this effect.
- The sensor and image sides use different gears and rigs. They are not paired samples, so there is no fusion. An image unit is one tooth crop; a sensor unit is one recording. The numbers sit side by side, not as one combined score.
- The Grader's slewing-bearing route and its checkpoints are not replaced. None of this is a field result from a motor grader.

## Deployment decision · November 8, 2026

Aalto is the primary gear-sensor candidate. The saved confusion matrices reproduce the published accuracy, balanced accuracy and macro F1, and the implementation groups outer and inner folds by physical gear pair. Each outer fold fits its own model. The experiment exports scores and figures without exporting a final inference estimator or its selected feature configuration. Neither the extracted Aalto data nor the cached features were available in the verification checkout. The MCC5-THU archive was also absent. The evaluations were reviewed from saved artifacts and were not rerun.

The next step is to obtain the verified Aalto archive or its provenance-checked feature cache and representative raw CSVs. Select the final feature set and estimator using grouped validation, fit the deployment artifact separately, and save its class order, feature order, channel schema, sampling rate and window configuration. Verify predictions using the exact Aalto preprocessing before adding a source-confirmed route. Treat the existing outer-fold score as an estimate for the model-selection procedure. A separate unseen gear pair would be needed to independently validate the final all-data artifact. Keep the JUST bearing route available as a separate diagnostic task.

The binary crosswalk is `healthy` to `good_tooth` and `tooth_damage` to `damaged_tooth` at the condition-category level. A good tooth does not certify the full gear as healthy. Aalto and the image dataset contain different physical gears. Same-grader field evidence remains an open requirement. See the [eight-equipment evidence matrix](equipment_evidence_matrix.md).

## Deployment model · November 10, 2026

The step above is done. `python Notebook/pdm.py grader-gear train-final` builds `Processed_Data/models/grader_gear_sensor.joblib` from all 425 Aalto runs of the 8 gear pairs. It is saved next to the Grader slewing-bearing model and does not replace it.

- **Selection:** the same three feature sets and five candidate models, chosen by macro F1 with grouped validation over gear pairs (4 inner splits, 2 pairs held out each time). Chosen: logistic regression on acceleration + shaft orders + torque (100 features). Its validation macro F1 is 0.892, but this score was used to make the choice, so it is optimistic. **The accuracy estimate stays the leave-one-gear-pair-out result above (77.8% balanced, 76.6% macro F1).** No gear pair is left to test the final fit on its own.
- **Saved with the model:** class order, feature order, the CSV columns (`time, enc1_ang, acc1..acc4, Torq1, Torq2`), 3012 Hz, 30 windows of 1 s after the first 10 s, mean window probability, the source and its licence, and the caveats.
- **Route:** an original Aalto 3012 Hz CSV is recognised by its columns and sent to this model by `predict`, the upload page and the integrated notebook. The answer says it is the gear-tooth task, not the slewing bearing. It has its own SHA-256 in the local trust list.
- **Smoke test (training data, so not an accuracy test):** `GP1.csv` (healthy, 250 rpm) → healthy 0.987; `Mild_Micropitting_GP4.csv` (1000 rpm) → tooth damage 1.000. `route-check` passed all 39 routes, including the Grader tooth-photo route.
- **Still open:** independent validation on a gear pair the final model has never seen, and any field data from a motor grader. The installation caveat above applies to this model too.

