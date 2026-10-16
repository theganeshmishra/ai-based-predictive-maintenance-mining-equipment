# Week 6: professor's remarks — 4 machines, 4 classes, CSV sensor models and CNNs

**October 2026 — Saskatchewan Polytechnic PROJ 613.** Remarks received after the Week 5 review:

1. Use **4-class classification**.
2. The **CSV (sensor) model should reach 90%+ accuracy**.
3. A **CNN on the images should reach about the same accuracy**.
4. Use **data augmentation**.
5. **Name each image by its machine** (for example `pump_...`, `belt_...`) so every image has a separate name.

The team also set the goal of **four machines ready by Week 5**; the Excavator was added as machine 4.

## What was done

| Remark | Implementation |
|---|---|
| 4 classes | Pump: healthy / inner / outer / ball. Haul Truck: healthy / inner / outer / **ball** (10 more UODS files, bearings 11–15). Belt Conveyor: healthy / wear25 / wear50 / wear75. **Excavator (new, machine 4):** healthy / bearing defect / unbalance / misalignment, from the CORA rig's other recorded faults. |
| CSV sensor model | `python Models/<machine>.py csv` writes `Processed_Data/<machine>_sensor_features_4class.csv` and trains models on it. The feature set and model (random forest, extra trees, gradient boosting, SVM, logistic regression) are **chosen on validation data inside the training set only**, then tested once. |
| Named images | `python Models/<machine>.py export-images` saves PNGs to `Images/<machine>/<class>/`, e.g. `pump_inner_20Hz_15_-30_t07.png`, `excavator_unbalance_F15_f123.png`, with an `index.csv` (local only, git-ignored). |
| CNN + augmentation | `cnn` (small CNN from scratch) and `cnn-pretrained` (ImageNet-pretrained **ResNet18**, transfer learning). Augmentation on training images: flip, shift, brightness/contrast, noise. Training length chosen on validation recordings. |
| New data | Haul Truck ball faults (Mendeley, SHA-256 checked). CORA vibration (Vx) for the Belt Conveyor and Excavator and CORA infrared for the Excavator: the CORA repository now shows a "not a bot" check, so the 40 files were downloaded in a browser and every file was checked against the publisher MD5 (`intake` / `intake-vibration` commands). |

## Results (accuracy; whole recordings or runs never seen in training)

| Machine | Sensor (CSV) | Images (CNN) | Test | 90% on both? |
|---|---:|---:|---|---|
| **Pump** | **0.958** (23/24) | **1.000** (24/24, pretrained ResNet18) | separate recordings / speeds | ✅ |
| **Haul Truck** | 0.867 (26/30) | 0.417 (5/12) | separate bearings / unseen speed | ❌ |
| **Belt Conveyor** | 0.625 (10/16) | 0.562 (9/16) | unseen speed | ❌ |
| **Excavator — main test** | 0.188 (3/16) | 0.312 (5/16) | unseen speed | ❌ |
| **Excavator — time split (secondary)** | **0.938** (15/16) | **0.938** (15/16) | later part of each run | ✅ (easier test, see below) |

Majority-class controls: Pump 0.250, Haul Truck 0.500 (sensor) / 0.250 (camera), Belt Conveyor and Excavator 0.250.

## How to read these results

- **Pump** meets the target on both sensor and images under the strict test.
- **Haul Truck:** the sensor model is close (0.867); most errors are outer-race faults. Ball-fault recordings were made **without load** (load sensor reads zero) while the others ran at about 400 N, so "ball" may partly mean "no load". The camera stays weak: the event images look almost the same for every fault, and a validation check of the camera's motion rhythm was also at chance, so it was not run on the test.
- **Belt Conveyor and Excavator (CORA rig):** CORA recorded each fault **once per speed**. The fair test holds out a whole speed, so the model must recognise a fault at a speed it never trained on, from three example runs. Adding vibration and per-speed normalisation raised the Belt Conveyor sensor model from 0.375 to 0.625, but neither machine reaches 90% this way.
- **Excavator time split:** training on the first 60% of every 45-minute run and testing on the last 30% (10% gap) gives 0.938 for both sensor and images. This shows the condition of a machine **already seen** can be recognised later in time. Because each fault comes from one recording, part of this score may come from recognising the recording rather than the fault, so it is reported **separately and labelled**, not as the main result.

No setting was chosen by looking at test results. Where an approach failed on validation data (camera motion rhythm for the Haul Truck), it was not tried on the test.
