# Week 6: professor's remarks — 5 machine mappings, 4 classes, CSV sensor models and CNNs

**October 2026 — Saskatchewan Polytechnic PROJ 613.** Remarks received after the Week 5 review:

1. Use **4-class classification**.
2. The **CSV (sensor) model should reach 90%+ accuracy**.
3. A **CNN on the images should reach about the same accuracy**.
4. Use **data augmentation**.
5. **Name each image by its machine** (for example `pump_...`, `belt_...`) so every image has a separate name.

The team also set the goal of **four machines ready by Week 5**; the Excavator was added as machine 4. The Dozer was added as machine 5 in Week 6 by mapping its final drive to the **same CORA gear-wear runs** already used for Belt Conveyor. It is not a fifth independent dataset.

## What was done

| Remark | Implementation |
|---|---|
| 4 classes | Pump: healthy / inner / outer / ball. Haul Truck: healthy / inner / outer / **ball** (10 more UODS files, bearings 11–15). Belt Conveyor: healthy / wear25 / wear50 / wear75. **Excavator (machine 4):** healthy / bearing defect / unbalance / misalignment, from the CORA rig's other recorded faults. **Dozer (machine 5):** healthy / wear25 / wear50 / wear75, reusing the Belt Conveyor CORA runs. |
| CSV sensor model | `python Models/<machine>.py csv` writes `Processed_Data/<machine>_sensor_features_4class.csv` and trains models on it. The feature set and model (random forest, extra trees, gradient boosting, SVM, logistic regression) are **chosen on validation data inside the training set only**, then tested once. |
| Named images | `python Models/<machine>.py export-images` saves PNGs to `Images/<machine>/<class>/`, e.g. `pump_inner_20Hz_15_-30_t07.png`, `excavator_unbalance_F15_f123.png`, with an `index.csv` (local only, git-ignored). |
| CNN + augmentation | `cnn` (small CNN from scratch) and `cnn-pretrained` (ImageNet-pretrained **ResNet18**, transfer learning). Augmentation on training images: flip, shift, brightness/contrast, noise. Training length chosen on validation recordings. |
| New data | Haul Truck ball faults (Mendeley, SHA-256 checked). CORA vibration (Vx) for the Belt Conveyor and Excavator and CORA infrared for the Excavator: only the needed CORA files were downloaded, and every file was checked against the publisher MD5 (`intake` / `intake-vibration` commands). |

## Results (accuracy, 4 classes)

**Main test** (fair): whole recordings, bearings or speeds that were never seen in training.
**Time split** (secondary, easier): the first 60% of every recording trains and the last 30% tests (10% gap), so the same recordings appear in training.

| Machine | Sensor, main | Images, main | Sensor, time split | Images, time split |
|---|---:|---:|---:|---:|
| **Pump** | **0.958** (23/24) | **1.000** (24/24) | — | — |
| **Haul Truck** | 0.800 validation-selected (24/30); fixed random forest 0.867 (26/30) | 0.417 (5/12) | **1.000** (30/30) | **0.913** (21/23) |
| **Belt Conveyor** | 0.625 (10/16) | 0.562 (9/16) | **1.000** (16/16) | **0.812** (13/16) local rerun; 1.000 saved earlier |
| **Excavator** | 0.188 (3/16) | 0.312 (5/16) | **0.938** (15/16) | **0.938** (15/16) |
| **Dozer*** | 0.625 (10/16) | 0.562 (9/16) | **1.000** (16/16) saved | **1.000** (16/16) saved |

Images = pretrained ResNet18 with data augmentation. Majority-class controls: Pump 0.250; Haul Truck 0.500 (sensor) / 0.250 (camera); Belt Conveyor and Excavator 0.250. The Haul Truck image time split has 23 test recordings because one ball recording is too short to have a later part.

\*Dozer repeats the Belt Conveyor's 16 CORA gear-wear experiments. Its matching scores are a second equipment-applicability mapping, **not** a new independent validation or data collection. See [the Dozer note](week_06_dozer.md).

**Local reproduction check (October 22):** The 16 CORA gear-wear vibration runs and all 28 Excavator source files were checked against the publisher's size and MD5 records. With the local environment rebuilt, the main validation-selected strict sensor results reproduced for Belt Conveyor (0.625), its shared-run Dozer analogue (0.625), and Excavator (0.188). The fixed random-forest Excavator result changed from 0.250 to 0.188 on this rerun; it is a secondary model and should not be described as exactly reproduced. The camera models were not retrained in this check. Raw and processed files remain local and Git-ignored.
The Haul Truck's 30 selected UODS sensor files and XJTU event-camera archive also passed publisher checksum checks. Its four-class sensor rerun reproduced the saved validation-selected accuracy (0.800), and 463 named local event images were exported. The Haul Truck camera model was not retrained; its saved image score remains a prior result.

**October 23 Belt Conveyor time-split rerun:** using the existing GitHub commands and locally cached inputs, `csv --split time` reproduced sensor accuracy 1.000 (16/16). `cnn-pretrained --split time` chose one epoch as before but returned image accuracy **0.8125 (13/16)** and macro F1 **0.8247**, whereas the earlier saved image result was 1.000 (16/16). The exact cause of this difference is not established. Both are *within-recording* time-split results; neither changes the separate whole-speed image result of 0.5625 (9/16). Do not cite the earlier perfect image score as exactly reproduced. The Dozer's saved image time-split score reuses the same CORA data and was not independently rerun here.

The other Week 6 **sensor** time-split commands also reproduced their saved accuracies locally: Haul Truck 1.000 (30/30), Excavator 0.9375 (15/16), and Dozer 1.000 (16/16). Pump's original held-out sensor result was already completed and was not rerun for this checkpoint. Haul Truck, Excavator and Dozer camera models were not retrained in this check, so their camera time-split figures remain saved prior results.

## How to read these results

- **Pump** meets the target on both sensor and images under the strict test.
- **Haul Truck:** the fixed random forest reaches 0.867; most errors are outer-race faults. Offering the validation step longer (0.5 s) envelope windows and the microphone channel did not help on the held-out bearings (0.800), and both results are reported rather than keeping the better one after seeing the test. Ball-fault recordings were made **without load** (load sensor reads zero) while the others ran at about 400 N, so "ball" may partly mean "no load". The camera stays weak: the event images look almost the same for every fault, and a validation check of the camera's motion rhythm was also at chance, so it was not run on the test.
- **Belt Conveyor and Excavator (CORA rig):** CORA recorded each fault **once per speed**. The fair test holds out a whole speed, so the model must recognise a fault at a speed it never trained on, from three example runs. Adding vibration and per-speed normalisation raised the Belt Conveyor sensor model from 0.375 to 0.625, but neither machine reaches 90% this way.
- **Preprocessing limitation:** the four-class CORA per-speed normalization uses unlabeled measurements from all runs at the held-out speed. This is batch-transductive, not a fully isolated new-machine prediction; the selected Belt/Dozer scores and some Excavator folds depend on it. No test labels were used, but a training-only preprocessing evaluation remains necessary before claiming deployment-style generalization.
- **Time split (the four non-Pump mappings):** training on the first 60% of each recording and testing on the last 30% (10% gap) gives 0.8125 to 1.000 in the saved results and latest Belt rerun. This shows the condition of a machine **already seen** can be recognised later in time. Because each fault comes from one recording, part of this score may come from recognising the recording rather than the fault, so it is reported **separately and labelled**, not as the main result. Dozer and Belt Conveyor share the same source runs.

For the original Week 6 models, settings were chosen without looking at their test results. Where an approach failed on validation data (camera motion rhythm for the Haul Truck), it was not tried on the test. Later exploratory checks on these already-used test runs do not count as a new confirmatory accuracy claim.
