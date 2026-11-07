# Grader with matched classes: plan (written before running)

PROJ 613 · Group 9 · November 7, 2026 · committed before any of these results existed

## Why

The Grader sensor model classifies a slewing bearing (healthy, inner ring, outer ring, rolling element). The Grader images are gear-tooth photos (good vs damaged tooth), so the two sides do not share classes. Web searches on November 5 and 7 found no public, licensed photo or thermal-image set with the four bearing classes (see the [image investigation](motor_grader_image_investigation.md)). The other way to match the classes is to change the sensor side to the image side's task. The motor grader circle drive is turned by a pinion gear, so **healthy gear vs damaged tooth** is a mechanically relevant shared task.

The archived Grader results and the deployment checkpoints stay unchanged. The new results go to `Results/experiments/grader_gear/`.

## Classes on both sides

| Side | healthy | damaged tooth |
|---|---|---|
| Images (existing, Roboflow GearBox_adddingImg v1) | `good_tooth` crops | `damaged_tooth` crops (broken, damaged, pitting and wear labels) |
| Sensor 1: MCC5-THU (already local) | `health_*` (12 recordings) | `teeth_break_L/M/H_*` (36) and `miss_teeth_*` (12) |
| Sensor 2: Aalto Gear Failure Dataset 3012 Hz v2 (Mendeley fywnj597d8, CC BY 4.0, 9.26 GB) | healthy baseline runs | runs with tooth damage (fracture, pitting or wear), as labelled by the source |

Compound faults (tooth break with a bearing fault), gear wear, pitting and crack are left out of MCC5-THU: the first are not single-fault, and the rest are the Drill Rig's classes.

## Test

- **MCC5-THU:** the Drill Rig's fair test, holding out 1000, 2000 and 3000 rpm in turn (4 healthy and 16 damaged recordings per round). The same 1 s windows and the feature sets `vibration`, `orders` and `vibration+orders` from attempt A. Feature set and model are chosen inside the training speeds with the existing five candidates (class-balanced). One prediction per recording.
- **Aalto:** after extraction, the folder and file structure will be recorded first. The held-out group will be the gear pair if the files identify it, otherwise the run or session. The baseline and damaged runs of one pair stay together. The same window features (without the tachometer orders unless an encoder speed is available), the same candidates, one prediction per run. Any change to this plan forced by the file structure is written into the results file before modelling.
- **Headline metric:** balanced accuracy, because damaged recordings outnumber healthy ones. Also accuracy, macro F1, the confusion matrix and the majority control.
- **Images:** the existing good vs damaged tooth result (balanced accuracy per tooth crop) is reused, not retrained.

## What can be claimed

The two sides would then answer the same question (is a gear tooth damaged?) on different gears and rigs. They are not paired samples, so there is no fusion and no claim that the sensor and the camera saw the same gear. MCC5-THU shares its rig and healthy recordings with the Drill Rig, so it is not independent of that result; Aalto is. The Grader's bearing route is not replaced, and none of this is a field result.
