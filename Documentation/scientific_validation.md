# Scientific validation

AI-Based Predictive Maintenance for Mining Equipment · Ganesh Mishra, Lakshya Sharma and Sagar Patel · November 11, 2026

**Readiness · Ready With Limitations.** The local application supports an academic demonstration of current component condition classification. The strongest aligned evidence is the Pump benchmark under changes in speed. CORA compares modalities on the same laboratory runs, with weak unseen-speed performance. The other mappings have different levels of component, label or test-unit alignment. Field validation, future-failure forecasting, and fused sensor–image inference remain outside the demonstrated scope.

The review used committed baseline `53ab430`, the [source manifest](../Dataset_Manifest/manifest.md), the [evaluation implementation](../Notebook/pdm.py), archived image provenance, both notebooks, and the saved result JSON. I recalculated sample counts, accuracy, balanced accuracy, macro precision, macro recall and macro F1 from 111 saved confusion-matrix blocks. Every corresponding saved value agreed within its rounding precision. This checks the arithmetic and recorded class support. It does not reproduce the predictions from raw data.

## Equipment, preprocessing and comparison scope

| Mapping | Sensor dataset and preprocessing | Image dataset and preprocessing | Primary outer test and comparison scope |
|---|---|---|---|
| Pump | Ottawa bearing vibration. 200 kHz, 100 ms windows with amplitude statistics and spectral bands. | XJTU-DV-Pump event camera. Polarity counts in 125 ms frames at 96 × 128, normalized per frame. | Sensor speed profiles A/B → C/D. Images 20 → 30 Hz. Both predict healthy, inner race, outer race and ball condition on separate rigs. Different speed protocols and unverified specimen IDs limit direct comparison. |
| Haul Truck | Ottawa UODS-VAFDC bearing vibration and microphone, 42 kHz. 100 ms statistics and spectral features, optional Hilbert envelope and microphone features. Long envelope uses 0.5 s windows. | XJTU-DV-Rotor. The same 125 ms event-frame representation. | Sensor holds out three bearing IDs per fold across five folds. Images hold out 2,000 rpm after training at 1,000 rpm. Four bearing labels align across separate rigs. The sensor ball class is confounded with load, and the image validation was sensitive to camera angle. |
| Belt Conveyor | CORA geared-drive Vx at 3 kHz. One-second windows sampled every ten seconds, statistics and frequency bands scaled by the drive setting. Optional T1 mean, standard deviation and slope from bounded samples. | CORA infrared, grayscale 96 × 128, all 450 frames per run. | Leave one drive setting out at F5, F15, F50 and F60. Same 16 runs and labels for both sides. Healthy and three wear levels. The same gear-wear setup may recur across speeds. |
| Dozer | Identical CORA Vx/T1 feature table to Belt Conveyor. | Every third infrared frame of those same runs, 150 per run. | Same test groups and classes as Belt Conveyor. The image sample is a subset of the conveyor export. These are two application mappings of one benchmark. |
| Excavator | CORA drivetrain Vx. Same Vx preprocessing as Belt Conveyor. | CORA infrared, 450 frames per run. | Four whole-setting folds, 16 shared sensor/image runs. Labels are healthy, bearing fault, unbalance and misalignment. Both modality results are at the 25% uniform four-class reference. |
| Wheel Loader | CORA electric-drive Vx. Same Vx preprocessing. | CORA infrared, 450 frames per run. | Four whole-setting folds, 16 shared sensor/image runs. Labels are healthy, bearing fault, half-broken and broken rotor bar. The healthy and bearing runs overlap Excavator (eight runs). |
| Drill Rig | MCC5-THU gearbox, six vibration channels at 12.8 kHz. One-second window statistics and spectral energy. | MFGNet-Gear synthetic meshes, two 224 × 224 renders per gear with ImageNet normalization. | Sensor leaves out 1,000, 2,000 and 3,000 rpm in turn. Images hold out three designs per fold across four folds. Healthy, wear and pitting broadly align. CAD root breakage and sensor tooth crack are a qualified crosswalk. |
| Motor Grader | JUST slewing bearing, six vibration channels at 50 kHz with one-second statistics and spectral energy. Aalto gear alternative uses 3,012 Hz, 30 one-second windows per run after 10 s, with acceleration statistics, encoder-based order bands and torque features. | Roboflow GearBox_adddingImg v1. Labeled tooth crops with 15% margin, 224 × 224, ImageNet normalization. | JUST holds out six conditions on four reused supports, one per class. Aalto holds out physical gear pairs. Images hold out filename-derived groups. JUST bearing labels and tooth labels describe separate tasks. Aalto and images share a binary damage concept, using different physical gears and different evaluation units. |

## Model settings and split audit

Sensor model and feature-set selection uses grouped validation within each outer training set, maximizing recording-level macro F1. The candidate settings in the audited code are Random Forest (250 trees, square-root feature sampling, minimum leaf 2, balanced subsampling), Extra Trees (300 trees, square-root feature sampling, minimum leaf 2, balanced class weights), histogram gradient boosting (200 iterations, learning rate 0.05, balanced class weights), standardized RBF SVM (C 10, gamma scale, balanced class weights, probability enabled), and standardized logistic regression (C 1, L-BFGS, 5,000 iteration limit, balanced class weights). The random seed is 42. SVM probability fitting happens within the outer training data and does not establish group-aware probability calibration.

The archived seven ResNet18 studies use ImageNet initialization and fine-tuning, AdamW with learning rate 0.0003 and weight decay 0.0001, and epochs selected by validation macro F1. Camera frames use class-weighted cross-entropy, flips, shifts, intensity scaling and noise, with one training recording per class reserved for validation. The Drill Rig uses unweighted cross-entropy, rotation, flips and shifts, three validation designs, and at most 50 training gears per design and class. The Grader images instead use a frozen ResNet18 encoder and standardized, class-weighted logistic regression with fixed C 0.1. The alternate Grader settings were selected within training source groups and are reported separately.

The evaluator keeps recording windows and frames together in the primary split and averages class probabilities at recording or gear level. Grader photos are scored per tooth crop with groups kept together by source. The outer grouping limits direct duplication across train and test within the defined unit. It does not rule out shared specimens across speeds, shared hardware across mappings, repeated views, or condition-related acquisition effects. The first seven image results came from pre-consolidation code at `c7d8d25`. A full reproduction with the current code remains unverified, especially for Drill Rig epoch selection. See the [reproducibility notes](../README.md#notes).

## Verified performance

These are percentages at the scoring unit shown. Macro precision, recall and F1 weight each class equally. All classes have test support in these rows, so macro recall equals balanced accuracy. An undefined class precision is scored as zero, consistent with the evaluator. Values absent from an archived headline block are derived from its confusion matrix and marked with an asterisk.

| Evaluation | Test units | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| [Pump sensor](../Results/pump/sensor.json) | 24 | 95.83% | 95.83% | 96.43% | 95.83% | 95.80% |
| [Pump image](../Results/pump/images.json) | 24 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| [Haul Truck sensor](../Results/haul_truck/sensor.json) | 30 | 80.00% | 73.33% | 76.04% | 73.33% | 74.10% |
| [Haul Truck image](../Results/haul_truck/images.json) | 12 | 58.33% | 58.33% | 64.58% | 58.33% | 56.55% |
| [Belt Conveyor sensor](../Results/belt_conveyor/sensor.json) | 16 | 43.75% | 43.75% | 33.33% | 43.75% | 35.00% |
| [Belt Conveyor image](../Results/belt_conveyor/images.json) | 16 | 50.00% | 50.00% | 41.79% | 50.00% | 43.18% |
| [Dozer sensor](../Results/dozer/sensor.json) | 16 | 43.75% | 43.75% | 33.33% | 43.75% | 35.00% |
| [Dozer image](../Results/dozer/images.json) | 16 | 56.25% | 56.25% | 56.25% | 56.25% | 52.08% |
| [Excavator sensor](../Results/excavator/sensor.json) | 16 | 25.00% | 25.00% | 14.58% | 25.00% | 18.33% |
| [Excavator image](../Results/excavator/images.json) | 16 | 25.00% | 25.00% | 19.17% | 25.00% | 21.67% |
| [Wheel Loader sensor](../Results/wheel_loader/sensor.json) | 16 | 31.25% | 31.25% | 35.00% | 31.25% | 32.54% |
| [Wheel Loader image](../Results/wheel_loader/images.json) | 16 | 18.75% | 18.75% | 34.17% | 18.75% | 20.56% |
| [Drill Rig sensor](../Results/drill_rig/sensor.json) | 120 | 73.33% | 68.06% | 82.60% | 68.06% | 70.54% |
| [Drill Rig image](../Results/drill_rig/images.json) | 4,800 | 88.94% | 88.94% | 88.92%* | 88.94%* | 88.81% |
| [Grader JUST sensor](../Results/grader/sensor.json) | 120 | 84.17% | 84.17% | 86.24% | 84.17% | 84.70% |
| [Grader tooth image](../Results/grader/images.json) | 2,879 | 97.22% | 82.59% | 89.96% | 82.59% | 85.84% |
| [Grader Aalto gear sensor](../Results/experiments/grader_gear/aalto_sensor.json) | 425 | 78.35% | 77.78% | 75.97% | 77.78% | 76.58% |
| [Grader MCC5-THU gear sensor](../Results/experiments/grader_gear/mcc5_sensor.json) | 60 | 88.33% | 70.83% | 93.64% | 70.83% | 76.01% |

Pump and Haul Truck score recordings. CORA scores runs, Drill Rig scores recordings and CAD gear instances, and Grader scores runs or tooth crops. The image-side test counts at frame or render level are Pump 1,946, Haul Truck 1,392, Belt Conveyor 7,200, Dozer 2,400, Excavator 7,200, Wheel Loader 7,200 and Drill Rig 9,600. These totals count related images from the scored groups. Grader’s 2,879 image units are tooth crops from 55 inferred sources.

### Settings selected per outer fold

Sensor order follows the saved folds. RF means random forest, ET extra trees, HGB histogram gradient boosting, SVM RBF SVM, and LR logistic regression. Each image epoch count is taken from the archived result.

| Mapping | Sensor model / feature set | Image epochs or fixed model |
|---|---|---|
| Pump | ET / vibration | 2 |
| Haul Truck | ET / vibration+long envelope+microphone; HGB / vibration+long envelope; RF / vibration; HGB / vibration+long envelope; LR / vibration+long envelope | 1 |
| Belt Conveyor | LR / vibration+temperature; ET / vibration; RF / vibration+temperature; ET / vibration+temperature | 15, 6, 6, 1 |
| Dozer | LR / vibration+temperature; ET / vibration; RF / vibration+temperature; ET / vibration+temperature | 7, 11, 1, 2 |
| Excavator | SVM / vibration; HGB / vibration; LR / vibration; HGB / vibration | 4, 2, 9, 4 |
| Wheel Loader | ET / vibration; ET / vibration; ET / vibration; LR / vibration | 3, 3, 5, 2 |
| Drill Rig | LR / vibration; RF / vibration; HGB / vibration | 6, 4, 6, 6 |
| Motor Grader | LR / vibration; LR / vibration; LR / vibration; LR / vibration; LR / vibration; LR / vibration | Frozen encoder + LR, C 0.1 |
| Grader Aalto gear sensor | LR / acceleration+orders; ET / acceleration+orders+torque; ET / acceleration+orders+torque; ET / acceleration+orders+torque; SVM / acceleration+orders+torque; ET / acceleration+orders+torque; ET / acceleration; LR / acceleration+orders+torque | See Grader tooth image row |
| Grader MCC5-THU gear sensor | LR / vibration; HGB / vibration; LR / vibration+orders | See Grader tooth image row |

### Confusion matrices

Rows are actual classes, columns are predictions. Class order follows each saved file.

| Evaluation | Class order | Counts |
|---|---|---|
| Pump sensor | healthy, inner, outer, ball | `[[6,0,0,0],[0,6,0,0],[0,0,6,0],[0,1,0,5]]` |
| Pump image | healthy, inner, outer, ball | `[[6,0,0,0],[0,6,0,0],[0,0,6,0],[0,0,0,6]]` |
| Haul Truck sensor | healthy, inner, outer, ball | `[[14,1,0,0],[1,3,0,1],[1,0,3,1],[0,0,1,4]]` |
| Haul Truck image | healthy, inner, outer, ball | `[[3,0,0,0],[1,2,0,0],[0,2,1,0],[0,0,2,1]]` |
| Belt Conveyor sensor | healthy, wear25, wear50, wear75 | `[[2,0,0,2],[2,0,1,1],[2,0,1,1],[0,0,0,4]]` |
| Belt Conveyor image | healthy, wear25, wear50, wear75 | `[[1,0,1,2],[0,3,1,0],[0,0,4,0],[1,2,1,0]]` |
| Dozer sensor | healthy, wear25, wear50, wear75 | `[[2,0,0,2],[2,0,1,1],[2,0,1,1],[0,0,0,4]]` |
| Dozer image | healthy, wear25, wear50, wear75 | `[[1,1,1,1],[0,4,0,0],[0,1,3,0],[1,2,0,1]]` |
| Excavator sensor | healthy, bearing, unbalance, misalignment | `[[2,0,2,0],[1,0,2,1],[1,0,2,1],[2,0,2,0]]` |
| Excavator image | healthy, bearing, unbalance, misalignment | `[[0,1,2,1],[0,1,2,1],[0,2,1,1],[0,2,0,2]]` |
| Wheel Loader sensor | healthy, bearing, half_broken_bar, broken_bar | `[[1,0,2,1],[1,2,1,0],[2,0,1,1],[1,1,1,1]]` |
| Wheel Loader image | healthy, bearing, half_broken_bar, broken_bar | `[[1,1,0,2],[0,1,2,1],[0,2,1,1],[0,2,2,0]]` |
| Drill Rig sensor | healthy, gear_wear, gear_pitting, tooth_crack | `[[5,7,0,0],[0,36,0,0],[0,9,25,2],[1,13,0,22]]` |
| Drill Rig image | healthy, gear_wear, gear_pitting, tooth_crack | `[[950,107,44,99],[155,990,9,46],[64,7,1129,0],[0,0,0,1200]]` |
| Grader JUST sensor | healthy, inner_ring, outer_ring, rolling_element | `[[25,0,0,5],[0,25,0,5],[0,4,26,0],[0,5,0,25]]` |
| Grader tooth image | good_tooth, damaged_tooth | `[[2690,24],[56,109]]` |
| Grader Aalto gear sensor | healthy, tooth_damage | `[[108,34],[58,225]]` |
| Grader MCC5-THU gear sensor | healthy, tooth_damage | `[[5,7],[0,48]]` |

## Generalization, leakage and interpretation

- **Pump** has the strongest aligned fault comparison, with 23/24 sensor and 24/24 image recordings correct. Its two holdouts probe different operating changes. The available identifiers do not establish a specimen holdout. [Ottawa data descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC6249544/)
- **Haul Truck** uses bearing-grouped sensor folds but associates all ball-fault recordings with no load and the other classes with nominal 400 N. The load channel is excluded from model features, yet the signals can still encode load. Removing ball changes the task to three classes, yielding 84.0% accuracy and 73.3% balanced accuracy. On the visual side, the archived score was 7/12. The follow-up rerun scored 8/12, with no validation recordings correct during epoch selection in that rerun. That finding is consistent with scene or camera-angle dependence. Its cause was not isolated. [UODS-VAFDC publisher](https://data.mendeley.com/datasets/y2px5tg92h/1) · [Follow-up experiments](followup_experiments.md) · [Improvement B](improvement_results.md#b-haul-truck-images-frames-covering-the-same-number-of-turns)
- **CORA** provides the closest run-level comparison because the sensor and infrared sides share run IDs and fault labels. The app fits each modality independently. Shared run IDs alone provide neither verified frame-to-window alignment nor a measured fusion gain. I matched the local image indexes to vibration filenames and checked for run overlap across the speed split. The four mappings contain 36 distinct stationary runs after deduplication by run ID. Belt Conveyor and Dozer share all 16 runs. The two drive mappings also share four healthy runs with the gear-wear mappings and eight healthy/bearing runs with each other. A single error moves each 16-run score by 6.25 percentage points.
- **CORA follow-ups** answer narrower questions. Healthy-versus-worn balanced accuracy was 45.8% sensor and 75.0% infrared on 16 runs. The infrared exact calculation tests random assignment of healthy predictions conditional on the observed prediction count. It does not test improvement over the collapsed four-class model. The 97.5–100% transient results held out repetitions at known speeds and used a different acquisition protocol and preprocessing. They cannot isolate speed as the sole cause of weak stationary results. The high time-split values measure later portions of recordings seen in training. [Binary follow-up](followup_experiments.md) · [Transient follow-up](improvement_results.md#c-cora-transient-runs-unseen-repetition-at-a-known-speed)
- **Drill Rig** sensor data are imbalanced (12 healthy and 36 per fault). Healthy recall is 41.7% in the archived unseen-speed sensor test. The order and envelope follow-ups reached 77.5% and 75.0% accuracy. They reuse the same benchmark and are follow-up comparisons. The CAD result applies to the generated designs and render pipeline. No photo-domain performance or upper bound follows from that result. The class called `tooth_crack` in the image JSON is R0 root breakage. [Label crosswalk](followup_experiments.md#priority-4-drill-rig-label-crosswalk-and-pump-grouping)
- **Motor Grader** JUST conditions reuse four supports, with class and specimen confounded. For teeth, 2,714/2,879 crops are good. An always-good classifier would get 94.27% accuracy, so the observed 82.59% balanced accuracy and 66.06% damaged-tooth recall give more context than 97.22% accuracy alone. Filename grouping controls known source overlap without establishing physical gear independence. For an alternate export, matching counts and group totals does not establish matching image content. Aalto holds out gear pairs, while healthy and damaged conditions also differ by installation. MCC5-THU holds out speeds and reuses the Drill Rig source and its healthy recordings. [Gear source audit](motor_grader_gear_comparison.md) · [Matched-class results](grader_matched_classes_results.md) · [Aalto publisher](https://data.mendeley.com/datasets/fywnj597d8/2)

## Motor Grader deployment verification

The local Aalto checkpoint is a fresh fit on all 425 runs from the SHA-256-verified source archive. It uses standardized logistic regression on 100 acceleration, order and torque features. The selected model is registered in the local trust manifest. The exact 250 RPM `GP1.csv` returned `healthy` with **98.7% model probability** in the CLI, notebook and upload page. This recording was used for the final fit. For generalization, use the separate archived leave-one-gear-pair-out evaluation of the model-selection procedure (**78.35% accuracy, 77.78% balanced accuracy, 76.58% macro F1**). [Local rebuild and verification](coding_completion_audit.md#aalto-real-file-recovery-and-rebuild)

## Technical checks and deployment prerequisites

| Check | Result |
|---|---|
| Full local pytest | 53 passed, 0 failed, 0 skipped. Existing scikit-learn SVC deprecation warnings were emitted. |
| Lint and dependency compatibility | Pyflakes and `pip check` passed. |
| CLI route check | 39 prediction calls passed, including the Aalto route. Unsupported input and ambiguous CORA checks also passed. |
| Integrated notebook | Clean kernel passed all eight sensor and eight image mappings and 19 upload checks. A separate clean-kernel run passed the real Aalto recording. |
| Local upload server | 16 real-file uploads and six error or ambiguity checks passed. A separate multipart upload passed with the real Aalto recording and matched the CLI probability. |
| Research notebook | All 18 code cells executed in a clean kernel. |
| GitHub Actions | The [audited baseline workflow](https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment/actions/runs/42968287263) passed. The [main workflow page](https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment/actions?query=branch%3Amain) tracks documentation commits. Hosted CI skips real-file tests when the local model or recording is absent. |

For a second computer, provision the project environment, all 17 checkpoints, trusted hashes for the locally verified Joblib files, indexed image exports, and the raw or prepared samples listed in the [demo script](demo_script.md). The optional whole-machine photo detector also needs ONNX Runtime and its separate files. The verification computer has the deployment samples and the Aalto recording, archive and feature cache. The local Grader COCO export and checkpoint metadata match the evaluated archive SHA-256. Full MCC5-THU, JUST and MFGNet mesh archives are absent in this checkout. See the [dataset manifest](../Dataset_Manifest/manifest.md) for rebuild sources. The audit did not retrain archived models.

## Remaining work before academic evaluation

1. Use the primary held-out table for performance claims. Label time-split, follow-up, frame-level and deployment numbers with their own test units.
2. Rehearse the demonstration on the submission computer with the exact local checkpoints and samples. Keep the CORA ambiguity case and Aalto/Grader task distinctions visible.
3. Treat independent field specimens, matched grader data, synchronized fusion, and failure-time labels as future research. The software demonstration and laboratory comparison are ready within the evidence limits above.
