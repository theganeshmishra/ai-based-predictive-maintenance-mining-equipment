# Eight-equipment evidence matrix

PROJ 613 · Group 9 · November 5, 2026

The matrix describes eight equipment application mappings. The measurements come from laboratory analogues or component datasets, and the Drill Rig image benchmark uses synthetic CAD renders. Shared CORA run IDs support a comparison at run level. Frame-to-window synchronization was not verified. The Pump and Haul Truck modalities use separate rigs. Source links, licenses and paths are recorded in the [dataset manifest](../Dataset_Manifest/manifest.md).

## Components, labels and held-out groups

| Equipment | Sensor component and source | Image component and source | Fault classes and comparability | Sensor split | Image split | Main limitations |
|---|---|---|---|---|---|---|
| Pump | Support bearing. Ottawa bearing rig, vibration. | Support bearing housing. XJTU-DV-Pump event camera. | Healthy, inner-race, outer-race, ball fault. | Train speed profiles A/B, test C/D. | Train 20 Hz, test 30 Hz. | Aligned labels on separate rigs. Independent physical bearing IDs were not established for the image split. |
| Haul Truck | Wheel-motor bearing analogue. Ottawa UODS-VAFDC vibration and microphone. | Wheel-motor bearing housing analogue. XJTU-DV-Rotor event camera. | Healthy, inner-race, outer-race, ball fault. | Five outer folds by bearing ID. | Train 1,000 rpm, test 2,000 rpm. | Separate rigs. UODS ball faults were unloaded while other classes ran at about 400 N. Image outer-race and ball recall were both 33.3%. |
| Belt Conveyor | Geared drive analogue. CORA Vx vibration and optional bounded T1 temperature. | Geared drive analogue. CORA infrared frames from the same runs. | Healthy and 25%, 50%, 75% gear wear. | Leave one speed out across F5, F15, F50 and F60. | Same held-out speeds and run IDs as sensor. | One run per class and speed. Sensor wear25 recall and image wear75 recall were zero. Three sensor folds selected vibration plus temperature. |
| Dozer | Final-drive analogue. CORA Vx and optional bounded T1 from the Belt Conveyor runs. | Final-drive analogue. CORA infrared from the same runs. | Healthy and 25%, 50%, 75% gear wear. | Same 16 runs and speed folds as Belt Conveyor. | Same 16 runs and speed folds as Belt Conveyor. | All 16 run IDs are reused. Treat this as a second application mapping of the same evidence. |
| Excavator | Drivetrain analogue. CORA Vx vibration. | Drivetrain analogue. CORA infrared frames from the same runs. | Healthy, bearing fault, unbalance, misalignment. | Four whole-speed folds. | Same held-out speeds and run IDs as sensor. | Both accuracies are at the 25% uniform four-class reference. Local binary follow-ups answer a different question. |
| Wheel Loader | Electric-drive analogue. CORA Vx vibration. | Electric-drive analogue. CORA infrared frames from the same runs. | Healthy, bearing fault, half-broken rotor bar, broken rotor bar. | Four whole-speed folds. | Same held-out speeds and run IDs as sensor. | Eight healthy or bearing runs overlap Excavator. Image accuracy is below the 25% uniform reference and broken-bar recall is zero. |
| Drill Rig | Rotary-head reduction gear analogue. MCC5-THU gearbox vibration. | Reduction gear analogue. MFGNet-Gear synthetic CAD renders. | Healthy, wear and pitting align broadly. The sensor root-crack class is mapped in code to CAD R0 root breakage. | Hold out 1,000, 2,000 or 3,000 rpm in turn. | Four folds holding out three CAD designs each. | Root crack and root breakage require qualification. Synthetic render performance has no established transfer to camera photographs. |
| Motor Grader | Circle-drive slewing-bearing analogue. JUST vibration. | Small gear teeth from GearBox_adddingImg v1 real photographs. | Sensor: healthy, inner ring, outer ring, rolling element. Image: good tooth or damaged tooth. | Hold out one working condition across six folds. The same four supports recur across conditions. | Hold out filename-derived video/photo groups. Score tooth crops. | Different components, labels and evaluation units. One physical support per sensor class. Image damaged-tooth recall was 66.1%. |

The Drill Rig crosswalk follows the source definitions in [MCC5-THU](https://pmc.ncbi.nlm.nih.gov/articles/PMC11070670/) and [MFGNet-Gear](https://huggingface.co/datasets/rsmei/MFGNet-Gear). The Motor Grader acquisition and labeling evidence is detailed in the [bearing image investigation](motor_grader_image_investigation.md) and [gear comparison review](motor_grader_gear_comparison.md).

## Archived primary held-out results

Values are copied from the saved score blocks at baseline `05476ba`, rounded to two decimals. Accuracy is the fraction of correct evaluation units, balanced accuracy is the mean class recall, and macro F1 averages class F1 scores equally. A test unit is a scored record, run, render design instance or tooth crop, depending on the row. The count does not imply that many independent physical specimens.

| Equipment | Sensor test units | Sensor accuracy | Sensor balanced accuracy | Sensor macro F1 | Image test units | Image accuracy | Image balanced accuracy | Image macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [Pump](../Results/pump/) | 24 | 95.83% | 95.83% | 95.80% | 24 | 100.00% | 100.00% | 100.00% |
| [Haul Truck](../Results/haul_truck/) | 30 | 80.00% | 73.33% | 74.10% | 12 | 58.33% | 58.33% | 56.55% |
| [Belt Conveyor](../Results/belt_conveyor/) | 16 | 43.75% | 43.75% | 35.00% | 16 | 50.00% | 50.00% | 43.18% |
| [Dozer](../Results/dozer/) | 16 | 43.75% | 43.75% | 35.00% | 16 | 56.25% | 56.25% | 52.08% |
| [Excavator](../Results/excavator/) | 16 | 25.00% | 25.00% | 18.33% | 16 | 25.00% | 25.00% | 21.67% |
| [Wheel Loader](../Results/wheel_loader/) | 16 | 31.25% | 31.25% | 32.54% | 16 | 18.75% | 18.75% | 20.56% |
| [Drill Rig](../Results/drill_rig/) | 120 | 73.33% | 68.06% | 70.54% | 4,800 | 88.94% | 88.94% | 88.81% |
| [Motor Grader](../Results/grader/) | 120 | 84.17% | 84.17% | 84.70% | 2,879 | 97.22% | 82.59% | 85.84% |

The Grader image experiment uses balanced accuracy as its headline metric because 2,714 of 2,879 crops are labeled good. Its 97.22% raw accuracy is sensitive to that imbalance. The 4,800 Drill Rig synthetic gears represent 12 design groups. CORA has 16 test runs per row, so one changed decision moves pooled accuracy by 6.25 percentage points.

## Secondary time-split accuracy

These experiments use early segments for training and later segments from the same recordings for testing. They measure within-recording transfer and remain separate from the primary held-out results.

| Equipment | Sensor accuracy | Image accuracy |
|---|---:|---:|
| Haul Truck | 100.00% | 100.00% |
| Belt Conveyor | 100.00% | 100.00% |
| Dozer | 100.00% | 93.75% |
| Excavator | 93.75% | 100.00% |
| Wheel Loader | 93.75% | 100.00% |

## Local deployment checks

The [deployment report](deployment_verification.md) records successful inference for eight sensor and eight image routes. Its sample predictions and confidence values verify that the interface routes files and loads checkpoints. The smoke samples may have been included in a deployment fit. Some checkpoint training scopes also differ from the archived experiments. For example, the Belt Conveyor and Dozer upload models accept vibration alone, and the Drill Rig and Grader deployment fits use bounded training inputs. See the [20-day audit](proj613_20_day_audit.md) for the next experiment and its acceptance criteria.
