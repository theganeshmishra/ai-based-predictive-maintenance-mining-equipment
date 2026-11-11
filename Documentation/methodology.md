# Methodology

## Task

The project classifies current component condition using eight equipment mappings. Pump and Haul Truck compare aligned bearing labels on separate laboratory rigs. The four CORA mappings use one laboratory rig with shared run IDs between sensor and infrared modalities. The scores are reported independently at run level. Frame-to-window synchronization was not verified. Drill Rig evaluates vibration and synthetic gear renders independently, with a qualified root-crack to root-breakage label mapping. Motor Grader has a JUST slewing-bearing task and two gear tasks. Aalto labels a gear-pair run and the photo model labels a tooth crop. See the [evidence matrix](equipment_evidence_matrix.md) for the source and test unit of each result.

## Workflow (the same for every machine)

All code is in one file, [Notebook/pdm.py](../Notebook/pdm.py). Its sections follow these steps in order.

```text
raw files ──► 1a load sensor data ──► feature table (one row per time window)   ──► 3b sensor model ──┐
          └─► 1b load camera data ──► PNG images + index.csv (one row per image) ──► 3c image model  ──┤
                                                                                                      ▼
                     2  split: decides what is training and what is test           3a one answer per
                     4  MACHINES: which groups each machine holds out              recording, metrics
```

1. **Load.** Each raw recording is cut into fixed windows (100 ms or 1 s, depending on the dataset). Each window gets features computed from its own signal: statistics such as RMS, kurtosis and crest factor, and the share of energy in frequency bands. The stationary CORA feature extractor scales frequency by the nominal drive setting. The transient follow-up uses measured RPM. Images are exported from the raw camera files. No feature uses a label.
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
| Grader | JUST holds out six conditions (120 recordings). Aalto holds out eight gear pairs (425 runs). | Each filename-derived video/photo group held out in turn (tooth crops). |

Within each evaluation, model, feature-set and epoch selection use outer training data. Later experiments motivated by results on the same benchmark remain follow-ups.

**Time split (secondary, easier).** Every recording is cut in time: the first 60% trains, the last 30% tests, and the 10% gap in between is dropped. The same recordings appear on both sides. This protocol measures transfer to later portions of known recordings. Settings are fixed (random forest on the first feature set; ResNet-18 with the same epoch rule).

## Leakage controls

- The primary tests check that held-out recording or source IDs do not overlap training. Specimen independence is limited by what the source IDs establish.
- No normalisation uses statistics from test recordings. Images are each scaled on their own; the sensor models scale inside the training fold.
- The raw mean of the UODS accelerometer is not a feature, because inner-race files carry a DC offset.
- Seeds are fixed (sensor 42, images 42, gears and gear renders 613). PyTorch training on a multi-core CPU is not bit-for-bit repeatable, so an image rerun can differ slightly from the saved score.

## Deployment verification

The upload routes use locally fitted checkpoints. Some fits use different inputs from the archived evaluations. For example, Belt Conveyor and Dozer uploads use vibration alone, while some evaluation folds selected temperature features. Deployment smoke tests check successful inference and error handling. Consult the [deployment verification report](deployment_verification.md) for those tests. Saved `.joblib` models load only when their SHA-256 is on the local trust list. The Grader gear-vibration model (Aalto, healthy vs tooth damage) is chosen with grouped validation over gear pairs and fitted on all 425 runs; the separate leave-one-gear-pair-out experiment estimates generalization of the selection procedure. The rebuilt checkpoint predicted healthy at 98.7% model probability on the 250 RPM training recording. The [final report](final_report.md) summarises the whole method with a [diagram](../Figures/methodology_flow.png).

## Follow-up experiments (November 6)

The [follow-up record](followup_experiments.md) uses the same folds and methods with a changed target or subset, saves to `Results/experiments/` and leaves the four-class results and checkpoints unchanged: CORA healthy vs worn (one prediction per held-out run, majority control, four-class model read as healthy vs worn), the Haul Truck fair test without the load-confounded ball class, and a route check with checkpoint hashes.

## Phase 1 diagnostics and follow-up

The signal diagnostics report waveform, FFT amplitude, Welch PSD and optional constant-speed shaft orders using the recording’s acquisition metadata. They run separately from model preprocessing. The grouped CORA candidate, fixed Pump disturbance tests and inference benchmarks are documented in the [Phase 1 enhancement report](phase_1_enhancement_report.md).

## Limits

- No data comes from mining equipment. Each machine maps a mining component onto a laboratory component.
- Test sets are small. One recording moves accuracy by 6.25 percentage points on a 16-run test and 8.33 points on the Haul Truck’s 12-recording image test.
- The CORA rig has one run per class and speed. The Dozer reuses the Belt Conveyor's runs, so it is not separate evidence. Merging the wear levels into healthy vs worn did not give defensible improvement (see the follow-up record). On the CORA transient runs, a held-out repetition at a trained speed scores 97.5–100%. The transient and stationary tests differ in acquisition and preprocessing as well as speed exposure ([improvement results](improvement_results.md)).
- The Haul Truck ball faults were recorded without load and the other classes at 400 N, so ball recall may partly measure load.
- The Pump source gives no bearing IDs, so its test holds out speed profiles, not bearings.
- Drill Rig image performance applies to CAD renders. Camera generalization remains untested. The code maps CAD root breakage to sensor root crack. Keep this label qualification visible.
- The Grader images use a different method: frozen ImageNet ResNet-18 features and a logistic regression, with settings fixed in advance (choosing them inside the training sources is saved as a check). They show real photos of small gears, not the slewing ring, and are scored by balanced accuracy because 94% of the tooth crops are good teeth.
