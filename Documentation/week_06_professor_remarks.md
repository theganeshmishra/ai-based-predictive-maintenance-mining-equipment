# Week 6: professor's remarks and first results

**October 2026 — Saskatchewan Polytechnic PROJ 613.** Remarks received after the Week 5 review:

1. Use **4-class classification**.
2. The **CSV (sensor) model should reach 90%+ accuracy**.
3. A **CNN on the images should reach about the same accuracy**.
4. Use **data augmentation**.
5. **Name each image by its machine** (for example `pump_...`, `belt_...`) so every image has a separate name.

## What was done

| Remark | Implementation |
|---|---|
| 4 classes | Pump: healthy / inner / outer / ball (unchanged). Belt Conveyor: healthy / wear25 / wear50 / wear75 (the dataset's own wear levels). Haul Truck: added **ball fault** (10 more UODS files, bearings 11–15, publisher SHA-256 checked) to healthy / inner / outer. |
| CSV sensor model | `python Models/<machine>.py csv` writes `Processed_Data/<machine>_sensor_features_4class.csv` and trains a majority-class control, logistic regression and a random forest (main model) on it. |
| Named images | `python Models/<machine>.py export-images` saves PNGs to `Images/<machine>/<class>/`, e.g. `pump_inner_20Hz_15_-30_t07.png`, `haul_truck_ball_2000rpm_angle1_t03.png`, `belt_conveyor_wear50_F15_f123.png`, with an `index.csv`. Local only (git-ignored); the command recreates them. |
| CNN + augmentation | `python Models/<machine>.py cnn`: 4 convolution blocks, 20 epochs, Adam; augmentation on training images only (horizontal flip, shift up to 8 px, brightness/contrast ×0.8–1.2, Gaussian noise); each image standardised on its own. |

All models keep the same fair test as before: whole recordings (or runs) are held out, and settings were fixed before running. No setting was chosen by looking at test results.

## Results (accuracy on held-out recordings/runs, 4 classes)

| Machine | CSV sensor (random forest) | CNN on images | Majority-class control | Images |
|---|---:|---:|---:|---:|
| Pump | **0.958** (23/24) | 0.833 (20/24) | 0.250 | 960 |
| Haul Truck | 0.867 (26/30) | 0.250 (3/12) | 0.500 sensor / 0.250 camera | 463 |
| Belt Conveyor | 0.375 (6/16) | 0.312 (5/16) | 0.250 | 2,400 |

- **Pump:** the sensor model meets the 90% target. The CNN reaches 0.833; it misses four of six inner-race recordings (predicted as ball). The earlier OpenCV HOG + random forest model on the same images scored 1.000.
- **Haul Truck:** the 4-class sensor model reaches 0.867; most errors are outer-race faults (3/5 right). **Caveat:** the ball-fault recordings were made with no load (the load sensor reads zero) while all others ran at about 400 N, so "ball" may partly mean "no load". The CNN stays at chance (0.250), consistent with the event images looking almost identical across classes.
- **Belt Conveyor:** neither model separates the four wear levels (0.375 and 0.312). Only the temperature channel and infrared frames are available locally: the CORA repository now shows a "not a bot" check that blocks automated download of its vibration files, and we do not bypass it.

## Honest status against the targets

Only the Pump sensor model reaches 90% so far. Reaching it elsewhere needs better data or methods, not tuning on the test set:

- **Belt Conveyor:** the CORA vibration channels (VX/VY/VZ) are the most promising route. They would need to be downloaded by hand through the website.
- **CNNs:** try a pretrained CNN (transfer learning, as listed in the proposal), with epochs and settings chosen on a validation split taken from the training recordings only.
