# Eight-equipment condition-monitoring methodology

This is the Saskatchewan Polytechnic PROJ 613 **research prototype**, not a claim
that a single model has been validated on eight mining fleets. The complete
[source/label verification matrix](../Dataset_Manifest/manifest.md) is the
authority for dataset provenance, access, licence, sample counts, and confidence.

## Component-level scope

| Equipment category | Selected monitored component and target | Experiment status |
|---|---|---|
| Haul/Dump Truck | Electric-drive wheel-motor bearing; healthy/inner/outer fault | B1 sensor–event-camera sources verified; **no paired model result yet** |
| Excavator | Swing-drive gear wear; healthy/worn | Shared CORA gear/infrared source verified; no model yet |
| Dozer | Final-drive gear wear; healthy/worn | Same CORA source, not a separate dozer dataset; no model yet |
| Wheel Loader | Final-drive/transmission gear wear; healthy/worn | Same CORA source; no model yet |
| Drill Rig | Rotary-head reduction gear wear; healthy/worn | Same CORA source; no matched visual model yet; the PHM rock-drill result is **sensor-only and a different fault target** |
| Grader | Circle-drive gear wear; healthy/worn | Same CORA source with a large mechanical-transfer caveat; no model yet |
| Belt Conveyor | **Geared** drive-reducer wear; healthy/worn | Same CORA source; not belt-crack or idler diagnosis; no model yet |
| Pump | Support-bearing healthy/inner/outer/ball fault | Separate recording-held-out vibration and event-camera models complete at component level; see [Pump report](../Results/pump_sensor_visual_comparison.md) |

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
   of observation. Never upload the large TiNA, CORA, Ottawa, or XJTU media.
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
- Report Accuracy, macro Precision/Recall/F1, per-class support and metrics,
  and run-level confusion matrices. The Pump primary tests both have 24 test
  recordings, but their rigs and operating-condition shifts differ. Their
  accuracy values are **descriptive, not a controlled modality comparison**.
  Do not subtract them, infer that one modality is superior, or present a
  fusion score. The sensor and visual records are not synchronized or paired.
- Existing UCI hydraulic, ESPset, and PHM rock-drill baselines have their own
  group-aware tests and labels in `Results/`. They must not be inserted into
  the Pump four-class comparison or represented as visual-fault validation.

## Claim boundary

All completed models classify **current** condition. No evaluated dataset
links a sensor/image observation to a later repair outcome for the same mining
machine; therefore the work does not estimate failure time or remaining useful
life. No mining fleet, unseen physical pump, or eight-machine deployment test
has been completed. The Pump result satisfies a component-level sensor-plus-
visual comparison only under the professor's stated allowance for sufficiently
similar components measured on different physical rigs.
