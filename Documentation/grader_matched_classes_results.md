# Grader with matched classes: results

PROJ 613 · Group 9 · November 7, 2026 · [plan](grader_matched_classes_plan.md) committed before running (c92b686)

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

- Both sides now answer the same question: is a gear tooth damaged? The two sensor results come from different rigs, and both beat the majority control by a clear margin on held-out gears (Aalto) or speeds (MCC5-THU).
- **Aalto is the stronger sensor test.** It holds out whole physical gear pairs, has 425 runs, covers four fault types, and is independent of every other machine in the project. Its 77.8% balanced accuracy is close to the image side's 82.6%, but under 80%.
- **MCC5-THU** shares its rig and healthy recordings with the Drill Rig, so it is not independent. Healthy gears at an unseen speed are its weak spot, as in the Drill Rig result.
- On Aalto, each pair's healthy runs and fault runs are separate installations, and the data paper notes installation-dependent amplitude. The model may partly detect a re-installation rather than the damage. Holding out whole pairs prevents memorising a pair, but it does not remove this effect.
- The sensor and image sides use different gears and rigs. They are not paired samples, so there is no fusion. An image unit is one tooth crop; a sensor unit is one recording. The numbers sit side by side, not as one combined score.
- The Grader's slewing-bearing route and its checkpoints are not replaced. None of this is a field result from a motor grader.
