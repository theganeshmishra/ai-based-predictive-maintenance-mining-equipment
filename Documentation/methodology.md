# Eight-equipment condition-monitoring methodology

This is the Saskatchewan Polytechnic PROJ 613 **research prototype**, not a claim
that a single model has been validated on eight mining fleets. The complete
[source/label verification matrix](../Dataset_Manifest/manifest.md) is the
authority for dataset provenance, access, licence, sample counts, and confidence.

**Week 4 implementation boundary (October 5, 2026):** all eight equipment
categories are mapped, and the component/dataset matrix is complete. Only the
Pump and Belt Conveyor sensor-versus-image experiments are complete. The six
other categories are planned but have **no completed matched-modality
experiment**. No Equipment #3 modelling begins in Week 4; see the
[progress checkpoint](week_04_progress.md). After Week 4, the Haul/Dump Truck
was added as Equipment #3 (see the table below).

## Component-level scope

| Equipment category | Selected monitored component and target | Experiment status |
|---|---|---|
| Haul/Dump Truck | Electric-drive wheel-motor bearing; healthy/inner/outer fault | Equipment #3 (after Week 4): separate bearing-held-out vibration and event-camera models complete on two different laboratory rigs; vibration well above its control, event camera near chance; see [Haul Truck report](../Results/haul_truck/haul_truck_sensor_visual_comparison.md) |
| Excavator | Swing-drive motor/gearbox faults; healthy/bearing/unbalance/misalignment | Done in Week 6 on the CORA drive-fault runs (a different CORA task from gear wear); see [Week 6](week_06_professor_remarks.md) |
| Dozer | Final-drive gear wear; healthy/25/50/75% wear | Done in Week 6 on the **same 16 CORA gear-wear runs and 4 classes as the Belt Conveyor** (not a separate dozer dataset), so the scores equal the Belt Conveyor's; see [Week 6: Dozer](week_06_dozer.md) |
| Wheel Loader | Final-drive/transmission gear wear; healthy/worn | Planned: same CORA source; no loader experiment yet |
| Drill Rig | Rotary-head reduction gear wear; healthy/worn | Planned: same CORA source; no matched drill-rig visual experiment yet; the earlier PHM rock-drill benchmark (since removed) was **sensor-only and a different fault target** |
| Grader | Circle-drive gear wear; healthy/worn | Planned: same CORA source with a large mechanical-transfer caveat; no grader experiment yet |
| Belt Conveyor | **Geared** drive-reducer wear; healthy/worn | Separate T1-temperature and infrared models complete on the same 16 CORA runs with whole-speed held-out testing; weak scores below majority-class accuracy; see [Belt Conveyor report](../Results/belt_conveyor/belt_conveyor_sensor_visual_comparison.md) |
| Pump | Support-bearing healthy/inner/outer/ball fault | Separate recording-held-out vibration and event-camera models complete at component level; see [Pump report](../Results/pump/pump_sensor_visual_comparison.md) |

The six CORA mappings reuse **one** laboratory gearbox dataset. They must never
be counted as six independent model validations. The original Pump vibration
source has only one selected recording per class; its within-recording score is
an exploratory diagnostic, not generalization. The completed Pump sensor model
instead uses repeated recordings from a separate Ottawa bearing rig as a
mechanically similar component proxy. It is not direct mining-pump evidence.

## Acquisition and preprocessing

1. Keep publisher source files and all derived feature caches under ignored
   `Raw_Data/` and `Processed_Data/`. Record source URL/DOI, version, licence,
   file identity, hashes when provided, channel meanings, conditions, and unit
   of observation. Never upload the large CORA, Ottawa, or XJTU media.
2. Choose a **condition-monitoring component and fault label** before using a
   sensor or image source. Equipment-identification photographs cannot stand in
   for fault-labelled images. Exclude compound or confounded labels when they
   do not match the target; the Ottawa compound class and UODS ball class are
   examples documented in the matrix.
3. For the Pump vibration proxy, verify each of 48 Ottawa MAT files against
   its publisher SHA-256. Use accelerometer `Channel_1` at 200 kHz; encoder
   `Channel_2` and file names are not predictors. Split each 10-second
   recording into 100 non-overlapping 0.1-second windows and extract 16
   time/spectral features. Average window probabilities to **one prediction
   per source recording**.
4. For Pump event-camera data, read only the required bounded interval of each
   remote XJTU ZIP member. Convert the first two seconds into eight 250 ms
   polarity-count images, extract OpenCV HOG features, and average frame
   probabilities to **one prediction per source recording**. These are motion
   images of a bearing housing, not RGB photos of surface damage. The full
   ZIP-member CRC was not verified from the bounded reads.
5. For CORA, preserve the published experiment key (health condition, speed,
   stationary/transient regime). Do not join an infrared frame to a sensor
   window until their timestamps and acquisition alignment are verified.
6. For the Belt Conveyor geared-drive analogue, use the 16 selected CORA
   stationary runs (`H`, `W25`, `W50`, `W75` at `F5`, `F15`, `F50`, `F60`).
   Download only their 16 infrared MATLAB stacks and eight deterministic
   256-KiB ranges per original T1-temperature CSV, retaining 6,000 complete
   readings per range. The range sample is roughly 4.3% of T1 source bytes,
   not the full signal. Verify file IDs, expected names, structural validity,
   450 frames per stack, range responses, and finite sensor samples. Record
   local SHA-256 hashes; only `H_F5_S.mat` also has a publisher MD5 check.
   Summarize each modality to **one feature vector per source run**. Infrared
   pseudo-colour pixels are OpenCV image descriptors, not calibrated Celsius
   temperatures. Do not claim frame/window synchronization.

## Evaluation and leakage controls

- Predeclare split units at **whole run, bearing ID, pump ID, configuration,
  or later time** as the source permits. Never split adjacent windows or frames
  from one recording across training and test and call that an independent
  result. Fit any feature selection only on training groups.
- The Pump Ottawa sensor test trains on all 24 A/B speed-profile runs and tests
  on 24 separate C/D runs (six test runs per class). A four-fold held-out
  speed-profile analysis checks sensitivity. The Pump visual test trains on
  24 20 Hz runs and tests on 24 separate 30 Hz runs; a six-fold held-out
  camera-configuration analysis checks sensitivity. Fixed random forests,
  most-frequent controls and shuffled **run-label** controls are recorded in
  the result JSON files. Hyperparameters were not selected using test runs.
- The Belt Conveyor CORA comparison has four leave-one-speed-out folds. For
  each speed, all four condition runs at that speed are test data and the 12
  other runs are training data. No sensor range or infrared frame from a test
  run enters training; scaling is fitted only on training runs. Separate fixed
  class-balanced logistic regressions yield one prediction per run, with the
  same folds and labels for both modalities. Pool the 16 out-of-fold run
  predictions for Accuracy, macro Precision/Recall/F1, per-class performance,
  and confusion matrices, and show the 75%-accuracy most-frequent control.
  This tests speed transfer on one laboratory rig, **not** transfer to an
  unseen gearbox or mining conveyor.
- The later Week 6 **four-class CORA sensor baselines** use a different feature
  pipeline from that original two-class comparison. Their per-speed-normalized
  candidate features are calculated across all recordings at each speed,
  including the four held-out runs at the test speed. This uses no test labels
  and the model is still fitted on training runs only, but it is a
  **batch-transductive preprocessing step**. The Belt Conveyor and Dozer
  selected scores use it in every fold; the Excavator selected score uses it
  in some folds. Those scores should not be presented as a fully inductive
  prediction on a single new machine. A training-only preprocessing rerun is
  needed before claiming that stronger deployment-style result.
- Report Accuracy, macro Precision/Recall/F1, per-class support and metrics,
  and run-level confusion matrices. The Pump primary tests both have 24 test
  recordings, but their rigs and operating-condition shifts differ. Their
  accuracy values are **descriptive, not a controlled modality comparison**.
  Do not subtract them, infer that one modality is superior, or present a
  fusion score. The sensor and visual records are not synchronized or paired.
- Existing UCI hydraulic and ESPset baselines have their own
  group-aware tests and labels in `Results/`. They must not be inserted into
  the Pump four-class comparison or represented as visual-fault validation.
- **80% accuracy gate:** require more than 0.80 recording/run-level accuracy
  in **each** modality for an equipment pair, with the same comparable fault
  classes, independent held-out units and per-class results. Exactly 0.80 is
  not above this gate. A high frame/window score, a same-recording time split,
  or the best of several models chosen after seeing the held-out results does
  not qualify. The existing CORA speed folds and XJTU Rotor test recordings
  have already been examined repeatedly; any further tuning on those same
  units is exploratory until a fresh, untouched test collection is available.
  One CORA run set reused for several equipment analogues remains **one**
  experiment, not independent accuracy evidence for each machine.

## Claim boundary

All completed models classify **current** condition. No evaluated dataset
links a sensor/image observation to a later repair outcome for the same mining
machine; therefore the work does not estimate failure time or remaining useful
life. No mining fleet, unseen physical pump, or eight-machine deployment test
has been completed. The Pump result satisfies a component-level sensor-plus-
visual comparison only under the professor's stated allowance for sufficiently
similar components measured on different physical rigs.

The Belt Conveyor experiment also completes a component-level sensor-plus-
infrared comparison, conditional on accepting CORA's laboratory geared drive
as a sufficiently similar reducer. Its sensor and infrared accuracies are
0.625 and 0.500, both below the 0.750 majority-class control; this is **not**
evidence of a reliable fault detector, mine deployment, or repair prediction.
