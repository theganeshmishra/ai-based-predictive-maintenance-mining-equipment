# Methodology

## Task

The project classifies current component condition using seven equipment mappings. Pump and Haul Truck compare aligned bearing labels on separate laboratory rigs. The four CORA mappings use one laboratory rig with shared run IDs between sensor and infrared modalities. The scores are reported independently at run level. Frame-to-window synchronization was not verified. Drill Rig combines vibration with synthetic gear renders and needs a root-crack versus root-breakage qualification. See the [evidence matrix](equipment_evidence_matrix.md) for the source and test unit of each result.

## Workflow (the same for every machine)

All code is in one file, [Notebook/pdm.py](../Notebook/pdm.py). Its sections follow these steps in order.

```text
raw files ──► 1a load sensor data ──► feature table (one row per time window)   ──► 3b sensor model ──┐
          └─► 1b load camera data ──► PNG images + index.csv (one row per image) ──► 3c image model  ──┤
                                                                                                      ▼
                     2  split: decides what is training and what is test           3a one answer per
                     4  MACHINES: which groups each machine holds out              recording, metrics
```

1. **Load.** Each raw recording is cut into fixed windows (100 ms or 1 s, depending on the dataset). Each window gets features computed from its own signal: statistics such as RMS, kurtosis and crest factor, and the share of energy in frequency bands. CORA bands are measured in shaft orders (frequency divided by shaft speed), so they line up across speeds. Images are exported from the raw camera files. No feature uses a label.
2. **Split.** See the two tests below. In the primary evaluations, keep all observations within the defined outer group on one side. In the secondary time split, use disjoint portions of the same recording.
3. **Train.**
   - *Sensor:* five candidate models (random forest, extra trees, gradient boosting, RBF SVM, logistic regression) and the machine's feature sets. The best pair is chosen by macro F1 on validation groups held out inside the training data.
   - *Images:* an ImageNet-pretrained ResNet-18 is fine-tuned. The number of epochs (up to 15; up to 6 for gears) is chosen on a validation part of the training data, then the model is retrained on all training images.
4. **Score.** Each test recording (or gear) gets one answer: the class with the highest mean probability over its windows or images. We report accuracy and macro F1 per recording, with a confusion matrix. A majority-class control shows what always guessing the most common class would score.

## The two tests

**Primary held-out evaluation.** The group definition varies by experiment. A held-out speed or operating condition can still contain specimens used at other conditions.

| Machine | Sensor test | Image test |
|---|---|---|
| Pump | train speed profiles A/B, test C/D (24 recordings) | train 20 Hz, test 30 Hz (24 recordings) |
| Haul Truck | 5 rounds, each holding out 3 whole bearings (30 recordings) | train 1000 rpm, test 2000 rpm (12 recordings) |
| Belt Conveyor, Excavator, Dozer, Wheel Loader | 4 rounds, each holding out one speed (16 runs) | same |
| Drill Rig | 3 rounds, each holding out one speed (120 recordings) | 4 rounds, each holding out 3 gear designs (4,800 gears) |

Every choice (model, feature set, number of epochs) is made inside the training data only.

**Time split (secondary, easier).** Every recording is cut in time: the first 60% trains, the last 30% tests, and the 10% gap in between is dropped. The same recordings appear on both sides. This protocol measures transfer to later portions of known recordings. Settings are fixed (random forest on the first feature set; ResNet-18 with the same epoch rule).

## Leakage controls

- The primary tests check that held-out recording or source IDs do not overlap training. Specimen independence is limited by what the source IDs establish.
- No normalisation uses statistics from test recordings. Images are each scaled on their own; the sensor models scale inside the training fold.
- The raw mean of the UODS accelerometer is not a feature, because inner-race files carry a DC offset.
- Seeds are fixed (sensor 42, images 42, gears and gear renders 613). PyTorch training on a multi-core CPU is not bit-for-bit repeatable, so an image rerun can differ slightly from the saved score.

## Deployment verification

The upload routes use locally fitted checkpoints. Some fits use different inputs from the archived evaluations. For example, Belt Conveyor and Dozer uploads use vibration alone, while some evaluation folds selected temperature features. Deployment smoke tests check successful inference and error handling. Consult the [deployment verification report](deployment_verification.md) for those tests.

## Limits

- No data comes from mining equipment. Each machine maps a mining component onto a laboratory component.
- Test sets are small. One recording is worth 6 points of accuracy on a 16-run test and 8 points on the Haul Truck's 12 image recordings.
- The CORA rig has one run per class and speed. The Dozer reuses the Belt Conveyor's runs, so it is not separate evidence.
- Drill Rig image performance applies to CAD renders. Camera generalization remains untested. The code maps CAD root breakage to sensor root crack. Keep this label qualification visible.
