# Motor Grader gear-tooth comparison feasibility

November 5, 2026 · PROJ 613

## Decision

Gear-tooth damage is a mechanically relevant target for a Motor Grader circle-drive analogue. Caterpillar documents a circle-drive pinion for its motor graders. The public sources below support a candidate research design, but I could not establish a gear-level image test set with verified physical gear IDs and comparable fault labels. The existing Grader routes therefore remain separate experiments. The sensor route measures slewing-bearing condition and the image route inspects an individual gear tooth. [Caterpillar circle-drive pinion](https://parts.cat.com/en/catcorp/product/261-8554)

| Candidate | Fault coverage and data quality | License and evaluation groups |
|---|---|---|
| [Aalto Gear Failure Dataset 3012 Hz, v2](https://data.mendeley.com/datasets/fywnj597d8/2) | Real multichannel signals from a scaled maritime thruster rig. Includes healthy baselines and replicated tooth-flank fracture, pitting and wear. The authors report installation-dependent amplitude variation and missing runs. | CC BY 4.0. The data paper documents gear-pair IDs and pre-damage baselines for each pair. Keep all states and runs of each pair together and verify per-class pair counts before freezing splits. [Data paper](https://acris.aalto.fi/ws/portalfiles/portal/167426184/1-s2.0-S2352340924011338-main.pdf) |
| [MCC5-THU v2](https://data.mendeley.com/datasets/p92gj2732w/2) | Vibration, speed and torque on a two-stage gearbox with healthy, missing-tooth, wear, pitting, crack and break conditions. Compound bearing faults are separately labeled. The seeded faults and rig differ from a grader. | CC BY 4.0. Speed/load runs support condition holdout. The documented conditions alone do not establish independent physical gear replication for every class. [Data paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11070670/) |
| [GearBox_adddingImg v1](https://universe.roboflow.com/vjs-jek7v/gearbox_adddingimg) | 838 real source images with tooth boxes. The capstone uses 2,714 good and 165 damaged tooth crops. An audit of the local COCO export found only five pitting annotations across two inferred source groups and three wear annotations in one group. The damage class also includes generic damage and breakage labels. | CC BY 4.0. The existing evaluation uses 55 filename-derived source groups. They are not verified physical gear IDs. Twelve inferred source groups overlap between the publisher train and test folders. A good tooth crop also cannot establish that the whole gear is healthy. |
| [Industrial helical gear scans](https://www.mdpi.com/1424-8220/23/20/8541) | Real full-gear scans with no-defect and damaged-tooth cases. The paper reports 30 damaged-tooth test scans, 99 no-defect scans from one station and 95 no-defect scans from another. This is the stronger image design because it inspects the full gear. | The data require permission from the industrial partner. The article’s open license does not establish dataset access or reuse rights. |
| [Gear pitting vibration-and-vision study](https://doi.org/10.1016/j.rineng.2026.109849) | Vibration and camera images were acquired synchronously on a dedicated gear rig. | The authors identify the underlying data as confidential. |

## Candidate experiment before implementation

Start with Aalto vibration and real full-gear inspection scans, subject to access and label verification. Use **healthy gear** versus **gear with verified tooth damage**. Include only fault types covered and verified in both sources. A pitting-only or wear-only comparison would need more independently identified image gears than the current export documents. Exclude compound bearing faults, ambiguous annotations and incomplete image coverage from this benchmark.

1. Record physical gear IDs or gear-pair IDs, label evidence, available views, and permitted uses. Count independent groups per class. Hold out complete gears on the image side and complete gear pairs on the vibration side. Keep baseline and damaged recordings of the same pair together.
2. Compare a simple vibration baseline using gear-mesh harmonics and sidebands against a tree model. For images, compare HOG with a linear classifier against ResNet18 features. Aggregate windows or tooth views into one result per gear or gear pair. Choose preprocessing, hyperparameters and thresholds on training groups only.
3. Report class support, balanced accuracy, macro F1 and confusion matrices on each independent test set. Different rigs and specimens permit a comparison of two approaches on a common condition, with no sample pairing or fusion. The result would remain a laboratory analogue because gear geometry, speed, load, fault creation and image access differ from actual grader operation.

The current 8 sensor and 8 image routes and their checkpoints were left intact. No model training or replacement was performed in this feasibility review.
