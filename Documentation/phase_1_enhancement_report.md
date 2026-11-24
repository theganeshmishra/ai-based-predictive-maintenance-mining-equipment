# Phase 1 enhancement and verification report

AI-Based Predictive Maintenance for Mining Equipment · Ganesh Mishra, Lakshya Sharma and Sagar Patel

**Final verdict (November 11) · Phase 1 Complete With Scientific Limitations.** See the final verification section at the end. **Earlier verdict · Ready for Phase 2 With Limitations.** The local system is ready for application development. The remaining limits concern held-out generalization, input provenance, and the availability of full research archives.

## Baseline and changes

The work started from `044d428ab85bf46631fad34f5f5c2722481d42f7` on GitHub main. The baseline rerun had 53 passed tests, zero failures or skips, and 39 successful prediction checks. Pyflakes passed. The existing application already supported eight mappings, three prediction interfaces, separate Grader bearing and gear tasks, and checkpoint trust verification. [Coding audit](phase_1_enhancement_report.md#coding-completion-audit) · [Scientific validation](scientific_validation.md)

This phase adds separate signal diagnostics, a reproducible experiment runner, controlled robustness probes, and a two-entry cache for the weights-only CPU ResNet18 models. The cache invalidates on file identity, size or timestamp changes. The Pump extractor now rejects non-finite samples before feature calculations. Short MCC5-THU recordings produce a minimum-length error. The validation script, test fixtures and new results are separate from the archived evaluations and deployment fits.

## Equipment-wise model results

Cells report accuracy / balanced accuracy / macro F1, in percent. The experiment budget was one candidate procedure on the three distinct, weak stationary CORA sensor tasks. Dozer shares the Belt Conveyor benchmark. The procedure summarizes the 10th, 50th and 90th percentiles of each existing window feature except log RMS and scores one row per run. For each held-out drive setting, it chooses one of four fixed models by mean inner macro F1 across the remaining settings. Candidates are standardized logistic regression, standardized RBF SVM (C 1), Random Forest and Extra Trees (200 trees, minimum leaf 2, seed 42). There are four outer groups and 16 scored runs per task. The archived Belt/Dozer pipeline could use T1 temperature. The candidate uses Vx alone, so the comparison evaluates complete pipelines and cannot isolate the effect of the new aggregation.

| Mapping | Archived sensor | Archived image | Phase 1 sensor candidate |
|---|---:|---:|---:|
| Pump | 95.83 / 95.83 / 95.80 | 100.00 / 100.00 / 100.00 | Existing sensor reproduced for robustness |
| Haul Truck | 80.00 / 73.33 / 74.10 | 58.33 / 58.33 / 56.55 | — |
| Belt Conveyor | 43.75 / 43.75 / 35.00 | 50.00 / 50.00 / 43.18 | 43.75 / 43.75 / 42.08 |
| Dozer | 43.75 / 43.75 / 35.00 | 56.25 / 56.25 / 52.08 | Same CORA result as Belt Conveyor |
| Excavator | 25.00 / 25.00 / 18.33 | 25.00 / 25.00 / 21.67 | 31.25 / 31.25 / 32.14 |
| Wheel Loader | 31.25 / 31.25 / 32.54 | 18.75 / 18.75 / 20.56 | 31.25 / 31.25 / 32.34 |
| Drill Rig | 73.33 / 68.06 / 70.54 | 88.94 / 88.94 / 88.81 | — |
| Motor Grader | JUST 84.17 / 84.17 / 84.70. Aalto 78.35 / 77.78 / 76.58. | Tooth crops 97.22 / 82.59 / 85.84 | — |

All checkpoints are retained. Excavator gained one of 16 correct runs. Belt Conveyor tied on accuracy and gained 7.08 points of macro F1, while Wheel Loader tied on accuracy and slightly declined on macro F1. The small, reused benchmark provides limited evidence for promotion. The candidate scores are follow-up comparisons on a familiar benchmark. The score JSON includes confusion matrices, selected models, and class order. [Phase 1 evidence](../Results/experiments/phase_1/verification.json)

Other mappings were reviewed using the archived results, follow-ups and deployment checks. Haul Truck’s image error depends on a small test and possible scene effects. More candidate tuning on those same images would give weak promotion evidence. The Drill Rig sensor improvement remains the archived 77.5% order-feature follow-up and uses a distinct protocol from the deployment fit. A calibrated field confidence estimate would require representative independent data. No calibration claim is made.

## Advanced signal diagnostics

`signal_diagnostics` reports a waveform, mean-removed single-sided FFT amplitude, Hann-windowed Welch density, band-integrated power, RMS, population variance and standard deviation, skewness, excess kurtosis, crest factor, peak-to-peak, and a constant-RPM shaft-order axis. Supply the sampling rate and units from source metadata. Constant RPM gives orders as frequency divided by rotation frequency. Variable-speed order tracking requires a measured speed trace and a separate method.

A 2-unit, 50 Hz sinusoid at 1,500 rpm returned FFT amplitude 2, RMS √2, variance 2, and peak order 2 in synthetic tests. The real JUST example uses the first second of channel `AI 1 (m/s2)` at 50 kHz and a nominal 2 rpm. RMS was 0.001658 m/s², peak-to-peak was 0.016288 m/s², and the strongest FFT bin was 30 Hz. One second provides 1 Hz resolution, equivalent to 30 orders at that speed. The order axis cannot resolve 1× rotation here. Band-integrated Welch estimates and all reported moments are in the score JSON. [Diagnostic figure](../Figures/experiments/phase_1/just_signal_diagnostics.png)

## Robustness

The archived Pump Extra Trees model was refit with its fixed settings on profiles A/B. Only the 24 held-out C/D recordings received disturbances. The clean score reproduced the archived result. Noise used seed 613 and Gaussian standard deviation equal to 10% of recording RMS. The 5% removal is random sample deletion followed by concatenation. The wrong-rate case resampled to 180 kHz and still presented the signal to the 200 kHz extractor.

| Disturbance | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Clean | 95.83% | 95.83% | 95.80% |
| Gaussian noise | 95.83% | 95.83% | 95.80% |
| Half amplitude | 95.83% | 95.83% | 95.80% |
| Double amplitude | 91.67% | 91.67% | 91.88% |
| 5% of samples removed | 70.83% | 70.83% | 62.47% |
| First half of recording | 91.67% | 91.67% | 91.43% |
| 180 kHz signal interpreted as 200 kHz | 70.83% | 70.83% | 62.47% |

All disturbed Pump recordings returned a class. The schema cannot independently infer the sampling rate of an incorrectly prepared upload. Correct publisher rates and continuous samples matter. The results apply to this laboratory holdout. [Robustness figure](../Figures/experiments/phase_1/pump_sensor_robustness.png)

The deployment probes separately rejected short Pump and Drill Rig inputs, NaN, infinity, malformed CSV/MAT/PNG, an unsupported extension, JPEG compression and resizing of the indexed Pump frame, and missing and untrusted checkpoints. JPEG and resize change the format fingerprint or pixel constraints. These checks cover safe routing and error messages. They provide no image-accuracy estimate on transformed samples. Per-case outcomes and confusion matrices are in the [verification JSON](../Results/experiments/phase_1/verification.json).

## Cross-validation and reproducibility

The grouped candidate checks recording IDs and keeps each complete drive setting on one side. Standardization fits inside each training split. The scientific review independently recalculated 111 archived confusion-matrix blocks and documented which groups were held out. Pump speed profiles do not establish unseen-bearing performance. Haul Truck ball faults are load-confounded. CORA reuses runs across mappings and provides no verified frame-window synchronization or fused model. Drill Rig image evidence comes from synthetic CAD. Motor Grader uses separate JUST supports, Aalto gear pairs and tooth crops. Condition classification and confidence outputs do not estimate future failure or remaining useful life. [Scientific validation](scientific_validation.md)

## CPU inference benchmark

Measured on Apple Silicon macOS, Python 3.12.14 and PyTorch 2.14.1, with CPU mode and two PyTorch threads. One fresh process per route and revision, with one cold and five repeated calls on the same real file. Preprocessing was timed separately once per route. The first load call includes the Joblib trust check for sensors or the PyTorch weight read for images. RSS is peak process memory including imports. The small benchmark measures local runs and does not establish throughput under concurrent use.

| Route | Checkpoint | Cold after | Warm median before → after | First load after | Preprocessing after | Peak RSS after |
|---|---:|---:|---:|---:|---:|---:|
| Pump sensor | 2.25 MiB | 202 ms | 148 → 148 ms | 28.4 ms | 119 ms | 449 MiB |
| Pump image | 42.72 MiB | 1,585 ms | 205 → 10.7 ms | 18.6 ms | 0.46 ms | 525 MiB |
| Aalto sensor | 7.29 KiB | 332 ms | 316 → 315 ms | 0.91 ms | 313 ms | 596 MiB |

The Pump image loaded weights once across six calls after caching, compared with six loads before. The measured predictions, including rounded probabilities, agreed before and after for all three routes. The image cache trades bounded resident model memory for lower repeat latency. Batch inference remains outside this benchmark.

## Run and verify

From the repository root with the project environment active. The completed research runners were retired from the current tree on November 17, 2026. Their original implementations remain available as [validation source](https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment/blob/48d95c50ad615eb4caa7f779d471ec981ea23058/Notebook/phase1_validation.py) and [final verification source](https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment/blob/48d95c50ad615eb4caa7f779d471ec981ea23058/Notebook/phase1_final_verification.py) at the recorded revision. The saved experimental evidence is retained.

```bash
python -m pytest -q -rs
python -m pyflakes Notebook/predictive_maintenance.py tests
python -m pyflakes Notebook tests
python Notebook/predictive_maintenance.py route-check
```

The historical experiment runner defaulted to `Processed_Data/phase1/` for outputs. It needs the Ottawa recordings, CORA stationary Vx recordings, the JUST demonstration excerpt, trusted local checkpoints, indexed image exports and the Aalto demo CSV. The paired benchmark uses an isolated checkout of `044d428` with access to the same local data and the new benchmark runner. Raw files and fitted models remain local.

## Regression and remaining limitations

- Local verification returned **69 passed, 0 failed, 0 skipped**. The real Pump and Grader Aalto tests ran alongside synthetic and mocked tests. Pyflakes and `pip check` passed. Existing scikit-learn SVC deprecation warnings remain.
- **39 real-file prediction checks passed**. The CLI also rejected an unsupported format and returned ambiguity for an unconfirmed CORA file. The integrated notebook passed 19 upload checks and a separate real Aalto check. The local web server passed 16 real uploads, six failure or ambiguity checks, and an additional real Aalto multipart upload.
- `data-check` listed eight missing research inputs. These are the MCC5-THU archive, MFGNet mesh archive and six JUST condition archives. Local deployment files cover every prediction route. The application does not require the full research archives for the demo.
- Tracked paths and documentation were rechecked for obsolete course naming and assistant artifacts. The existing ignore rules cover the local assistant files. [GitHub Actions for main](https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment/actions?query=branch%3Amain) runs the expanded lint and test suite. Hosted CI skips real-file tests when the local fixtures are absent.

## Final Phase 1 verification (November 11, 2026)

**Verdict · Phase 1 Complete With Scientific Limitations.** Every archived sensor result reproduces exactly from the raw data, no leakage was found, all tests and prediction routes pass, and two deployment defects found during the checks were fixed. The limits are scientific: small held-out sets, laboratory analogues only, four CORA mappings that do not generalise to an unseen speed, uncalibrated probabilities, and strong sensitivity of some sensor models to gain, noise and sampling-rate errors.

The checks and the one new experiment were fixed in advance in the [final verification plan](phase_1_enhancement_report.md#phase-1-final-verification-plan), committed before they ran. Historical runner: [final verification source](https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment/blob/48d95c50ad615eb4caa7f779d471ec981ea23058/Notebook/phase1_final_verification.py). Results: [final_verification.json](../Results/experiments/phase_1/final_verification.json). Archived results and deployed checkpoints are unchanged.

### 1. Commits and environment

- Start: `481d2e5e9b94027fae4a0cba1e575f925511067c`, GitHub main after a history rewrite pushed from another computer. This computer's earlier commits had identical content and were moved onto it. The final commit is the one that adds this section.
- Verified on Windows 11 (build 26200), Python 3.11.9, NumPy 2.4.6, pandas 2.3.3, scikit-learn 1.9.1, PyTorch 2.14.1+cpu, seed 613. The earlier Phase 1 figures were measured on Apple Silicon macOS, Python 3.12.14.

### 2. Tests run in this verification

| Check | Result |
|---|---|
| `python -m pytest -q -rs` at the start (481d2e5) | 69 passed, 0 failed, 0 skipped |
| Same, final (6 consecutive runs) | **230 passed, 0 failed, 0 skipped** in each run |
| `python -m pyflakes Notebook tests` | clean |
| `python Notebook/predictive_maintenance.py route-check`, final | **39 of 39 routes passed**, 0 failed; unsupported file rejected; unmapped CORA file returned "Ambiguous Equipment" |
| `pip check` | no broken requirements |

The 161 new test cases are in `tests/test_application.py` (archived results and final regression section). 148 of them check that every saved confusion matrix in `Results/` agrees with its reported accuracy, balanced accuracy, macro F1 and unit count. The rest are leakage guards, cache bounds, analytic diagnostics checks and UODS trimming. They are synthetic or read committed result files, so they also run on GitHub. The real-file tests (Pump and Aalto routes) ran locally and are skipped on GitHub, where the data is absent.

### 3. GitHub Actions

The workflow passed on the starting commit 481d2e5 ([run 43059385040](https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment/actions/runs/43059385040)). The result for the final commit is on the repository's Actions page. GitHub runs lint and the synthetic tests only; the real-file checks above are local.

### 4. Defects found and fixed

- **Haul Truck `.mat` uploads** failed with an internal pandas error ("Length of values (95) does not match length of index (99)") unless the recording was an exact multiple of 21,000 samples (0.5 s). The CSV route already trimmed partial blocks. Both routes now share `uods_array_features`, which trims the same way and gives clear messages for short or non-finite recordings. All 30 training recordings are exact multiples and finite, so the archived features are unchanged, and the Haul Truck fair test reproduces exactly after the change.
- **Upload page refusals were sometimes lost on Windows.** A refused upload (for example one from another website) closed the connection with the request body unread. Windows then reset the connection, and the client could lose the 403 page. `test_uploads_from_other_websites_are_refused` failed in 3 of 24 full-suite runs. The server now reads and discards at most 1 MiB, for at most 1 s, before closing. After the fix the upload tests passed 25 of 25 times and the full suite 6 of 6. The refusal itself was always enforced; only the message could be lost.

### 5. Reproduction, uncertainty and calibration of the sensor fair tests

All eight sensor feature tables were rebuilt from the raw archives and the archived fair tests rerun. **All eight reproduced exactly**: identical recording-level confusion matrices and the same chosen feature set and model in every round.

How the table below was computed:
- Probabilities are the mean window probability per test recording.
- Intervals are Wilson 95% intervals on accuracy.
- ECE uses 5 equal-width bins. With 16 to 120 recordings it is descriptive only.

| Mapping | Test recordings | Accuracy (95% interval) | Brier | ECE | Mean confidence: correct / wrong |
|---|---:|---|---:|---:|---|
| Pump | 24 | 95.8% (79.8–99.3) | 0.08 | 0.07 | 0.92 / 0.68 |
| Haul Truck | 30 | 80.0% (62.7–90.5) | 0.28 | 0.06 | 0.89 / 0.54 |
| Belt Conveyor (= Dozer) | 16 | 43.8% (23.1–66.8) | 0.70 | 0.23 | 0.69 / 0.59 |
| Excavator | 16 | 25.0% (10.2–49.5) | 1.00 | 0.43 | 0.76 / 0.65 |
| Wheel Loader | 16 | 31.3% (14.2–55.6) | 0.82 | 0.17 | 0.53 / 0.43 |
| Drill Rig | 120 | 73.3% (64.8–80.4) | 0.41 | 0.14 | 0.67 / 0.55 |
| Motor Grader (JUST) | 120 | 84.2% (76.6–89.6) | 0.17 | 0.13 | 0.85 / 0.52 |

- Pump and Haul Truck are roughly calibrated on their held-out recordings, and their wrong answers carry clearly lower confidence.
- The CORA models are overconfident. The Excavator averages 68% confidence at 25% accuracy, and one of its wrong answers scored above 0.9.
- The Drill Rig and Grader are under-confident on average.
- The three CORA intervals all include the 25% control.
- The Haul Truck's 80.0% has a lower bound of 62.7%.
- **The "model probability" in predictions is the classifier's averaged class score, not a calibrated failure probability.** No calibration was fitted, because the held-out sets are too small to fit and test one. Deployed outputs were not changed.

### 6. Leakage and cross-validation audit

| Check | Result |
|---|---|
| Each recording has one class, one held-out group and one inner validation group (8 mappings) | yes |
| Each round holds out exactly its declared groups; no recording on both sides; each tested once | yes |
| Model and feature-set selection sees only training rows | asserted in every round of the rerun; new test with a spy on `choose_model` |
| Scaling fitted inside training only | `StandardScaler` sits inside the SVM and logistic-regression pipelines (new test); tree models are unscaled |
| Identical feature rows shared between recordings, or across train and test | 0 in all eight tables |
| Images: identical files (SHA-256) across each round's train and test | 0 in all 7 exported indexes (40,204 images); no unit crosses the split; every unit sits in one group |
| Saved confusion matrices against reported metrics | all 148 blocks agree (new CI test) |
| Model comparisons use one unit per recording, run or gear | yes; image results report per-image scores separately |

What this audit does not cover, and the known sharing between mappings:
- **Near-duplicate images:** images that are near-identical but not byte-identical aren't detected. Neighbouring video frames inside one recording stay on one side, because whole recordings are held out.
- **Shared datasets:** the Dozer reuses the Belt Conveyor runs. The Excavator and Wheel Loader share the healthy and bearing runs. The Drill Rig sensor and the MCC5 Grader follow-up share the MCC5-THU rig.
- **Grader tooth photos:** these are split by source set inside the archive and have no exported index, so they weren't part of the image hash check.

### 7. Weak mappings: diagnosis and experiment E

**What the CORA results show:**
- The confusion matrices show no single systematic confusion. Errors are spread across classes and change from speed to speed. Excavator recall, for example: healthy 2/4, unbalance 2/4, bearing 0/4, misalignment 0/4.
- Each class has one stationary run per speed, so each held-out speed tests one run per class, and one run moves accuracy by 6.25 points.
- Attempt C already showed that the features separate the classes at a speed seen in training (97.5–100%).
- The earlier Phase 1 CORA candidate was evaluated on the same 16 runs, so it is a follow-up comparison, not independent evidence.

**Experiment E (planned in advance, run once):** the transient start-up runs (5 repetitions per class and speed) tested at an unseen speed. There were four rounds of 20 test runs, 80 in all. The model was chosen inside the training speeds. The majority control is 25%.

| Mapping | Accuracy (95% interval) | Balanced | Macro F1 | Stationary fair test |
|---|---|---:|---:|---:|
| Belt Conveyor (= Dozer) | **63.7% (52.8–73.4)** | 63.7% | 63.3% | 43.8% |
| Excavator | 31.3% (22.2–42.1) | 31.3% | 31.2% | 25.0% |
| Wheel Loader | 35.0% (25.5–45.9) | 35.0% | 31.9% | 31.3% |

**What E shows:**
- With five runs per class and speed, gear wear (Belt Conveyor, Dozer) is recognised clearly above chance at an unseen speed. So its stationary failure is partly a data-quantity problem.
- The drive faults (Excavator, Wheel Loader) stay near chance. For these features, the change of speed itself is the barrier.
- Experiment E uses transient runs, which the deployed routes don't accept. So nothing is deployed, and the archived stationary results stand.
- Better data for these mappings means several stationary runs per class at each speed, from more than one specimen per fault.

### 8. Robustness beyond the Pump

This covers the Haul Truck, Drill Rig and Motor Grader (JUST):
- In each archived round, the archived chosen model was refitted on the training groups.
- Only the held-out recordings were disturbed, using the Pump disturbance set with seed 613.
- The clean case reproduced each archived confusion matrix exactly.

Each cell shows accuracy / balanced accuracy, then the share of recordings whose class didn't change.

| Disturbance | Haul Truck (30) | Drill Rig (120) | Motor Grader (120) |
|---|---|---|---|
| Clean | 80.0 / 73.3 · 100% | 73.3 / 68.1 · 100% | 84.2 / 84.2 · 100% |
| Gaussian noise, 10% of RMS | 76.7 / 71.7 · 93% | **49.2 / 41.0 · 60%** | 84.2 / 84.2 · 91% |
| Half amplitude | 80.0 / 73.3 · 90% | 76.7 / 69.4 · 85% | **29.2 / 29.2 · 33%** |
| Double amplitude | 80.0 / 73.3 · 93% | 65.8 / 63.2 · 86% | **36.7 / 36.7 · 29%** |
| 5% of samples removed | 80.0 / 73.3 · 97% | **45.8 / 38.2 · 56%** | 89.2 / 89.2 · 95% |
| First half only | 80.0 / 73.3 · 100% | 74.2 / 68.8 · 96% | 86.7 / 86.7 · 98% |
| 90% rate read as full rate | 80.0 / 73.3 · 97% | **42.5 / 35.4 · 41%** | 75.0 / 75.0 · 78% |

- **Grader (JUST)** depends on the absolute vibration level, because its features include RMS and peak in m/s². A sensor with a different gain or calibration would break it.
- **Drill Rig** depends on fine spectral detail. Added noise, dropped samples and a wrong sampling rate each cut accuracy to 42–49%.
- **Haul Truck** was stable under every disturbance.
- No recording was rejected. The routes can't detect a wrong sampling rate or a wrong unit by themselves, so uploads must use the publisher's rate and units. These are synthetic disturbances on laboratory data, not a field robustness claim.

**Deployment probes:** these feed in short, NaN and infinite signals, malformed CSV, MAT and PNG files, an unsupported extension, JPEG-compressed and resized images, and missing and untrusted checkpoints. All 13 gave the same outcome on this computer as in the committed Phase 1 run. Transformed images are rejected by the format check, so no image-accuracy figure under compression or resizing exists.

### 9. Signal diagnostics

- **New analytic tests:**
  - a square wave: crest factor 1, excess kurtosis −2, peak-to-peak 6;
  - a two-level signal with known skewness and kurtosis;
  - white noise: Welch band power sums to the variance within 3%, and sub-bands add exactly;
  - signals scaled by 10⁻⁹ and 10⁹ give identical shape statistics.

  They pass alongside the existing sinusoid, zero-signal and bad-metadata tests.
- **The reported JUST example** was recomputed from its original archive member (`condition1.zip`). RMS 0.001658 m/s², peak-to-peak 0.016288 m/s², strongest FFT bin 30 Hz and 1 Hz resolution all equal the report. Its time column gives a measured rate of 50,000 Hz.
- **Sampling rates:** Aalto measures 3,012.05 Hz from its time column (declared 3,012). The Ottawa (200 kHz), UODS (42 kHz), CORA (3 kHz) and MCC5-THU (12.8 kHz) files have no time column, so their rates rest on publisher metadata.
- No diagnostic feature was added to any deployed preprocessing.

### 10. Classification errors by mapping (archived fair tests)

| Mapping | Main errors | Practical meaning |
|---|---|---|
| Pump | 1 of 6 ball faults called inner (sensor); images all correct | strongest evidence; still one rig, and an unseen speed profile, not an unseen bearing |
| Haul Truck | sensor inner and outer recall 3/5 each; images: outer and ball 1/3 each | sensor usable as a laboratory screen; images unreliable |
| Belt Conveyor, Dozer | sensor wear25 recall 0/4, healthy 2/4; images: wear75 0/4 (Belt Conveyor) | near chance at an unseen speed |
| Excavator, Wheel Loader | most classes 0–2 of 4 | at chance; do not use for decisions |
| Drill Rig | sensor: healthy recall 5/12; 13 crack and 9 pitting recordings called gear wear; images: 155 wear renders called healthy | a "gear wear" answer is unreliable (precision 55%); images are CAD renders |
| Motor Grader | sensor: recall 83–87% per class; images: **56 of 165 damaged teeth called good (recall 66%)** | the tooth-photo model misses one damaged tooth in three, so a "good tooth" answer does not rule out damage |

### 11. Reproducibility

| Category | Items |
|---|---|
| Fully reproduced here, bit-identical | all 8 archived sensor fair tests; the Phase 1 CORA candidates and Pump robustness (identical scores and choices); the 13 deployment probes; the JUST diagnostic values |
| Reproducible with a caveat | image fair tests: PyTorch CPU training is not bit-for-bit repeatable (documented); not rerun |
| Partially reproducible | the archived validation run: its diagnostic figure reads `Processed_Data/deployment_samples/grader/20221208-O-2rpm-0N-1.csv`, which exists only on the computer that produced it; the same file is inside `condition1.zip` |
| Blocked by unavailable data | none on this computer: `data-check` found all 163 raw files; the only mismatch is the documented second Roboflow export of the gear photos |

The eight research inputs reported missing on the macOS computer (MCC5-THU, the MFGNet mesh archive and six JUST condition archives) are present here and pass `data-check`.

### 12. Inference reliability (Windows, CPU)

How this was measured:
- One fresh process per route.
- One cold call and five repeated calls on the same real file.
- Peak working set includes imports.
- Measured on this computer only; not a throughput test.

| Route | Cold | Warm median | Preprocessing | Checkpoint loads in 6 calls | Peak memory |
|---|---:|---:|---:|---:|---:|
| Pump sensor | 265 ms | 227 ms | 193 ms | 6 calls to the trusted loader; hashed and read once, then served from memory | 387 MiB |
| Pump image | 2,980 ms | 13.6 ms | 0.5 ms | 1 | 491 MiB |
| Aalto gear sensor | 372 ms | 348 ms | 345 ms | 6 | 363 MiB |

- **Cache behaviour:** the image cache holds at most two models, keeps routes apart, and reloads when a checkpoint file changes. A new test checks this by alternating three routes. Repeated predictions were identical.
- **Historical gap:** Image weights originally used weights-only loading without a registered hash. The hardening below closes this gap.

### 13. Missing datasets and checkpoints

- **This computer:** nothing is missing. Every checkpoint that `predict` uses is present, and the ten `.joblib` models pass their trust check.
- **Not available anywhere in the project:**
  - field data from mining equipment;
  - paired sensor and camera recordings of one component;
  - more than one stationary CORA run per class and speed.

### 14. Remaining scientific limitations

- Laboratory and synthetic analogues only; no mining-equipment validation. Condition classification only; no failure-time or remaining-useful-life prediction.
- Small held-out sets: 16 runs for each CORA mapping and 24 to 30 for the Pump and Haul Truck, giving wide intervals.
- The CORA drive-fault mappings don't generalise to an unseen speed, even with five runs per class.
- Probabilities are not calibrated.
- The Grader (JUST) is sensitive to sensor gain; the Drill Rig to noise, sample loss and sampling rate.
- Haul Truck ball faults are load-confounded. The Pump test holds out speed profiles, not bearings. The Drill Rig images are CAD renders. The Grader sensor and photo tasks concern different components.

### 15. Final verdict

**Phase 1 Complete With Scientific Limitations.** The acceptance criteria are met:
- the regression checks were rerun on this computer;
- all eight mappings and all 39 routes work;
- the weak mappings were investigated, with one pre-planned experiment;
- leakage was audited, with new automated guards;
- robustness testing was extended to three more mappings;
- the diagnostics were checked analytically and on real data;
- calibration limits are stated;
- reproducibility is documented;
- inference reliability was measured.

Phase 2 was not started.

## Final checkpoint hardening · November 13, 2026

Implementation baseline: `04e25d29d79219b2a5ad319c662d47d1a235cf50`.

All seven ResNet18 `.pt` classifiers (Pump, Haul Truck, Belt Conveyor, Dozer, Excavator, Wheel Loader and Drill Rig) now require a trusted SHA-256. The eighth image mapping, Motor Grader tooth inspection, uses the existing trusted Joblib loader. The optional whole-machine ONNX detector is a separate identification aid outside these component fault routes.

On this computer, actual model bytes matched the earlier local deployment digests before registration. On a cache miss, the exact bytes hashed are passed to PyTorch with `weights_only=True`. Each request checks the local registry and checkpoint file identity. The two-entry cache includes the registered hash, device, inode, size, modification time and change time. Bulk registration rejects changed entries. New checkpoints written by the training command are registered explicitly. Keep the local registry under operator control and verify provenance before registering models.

The local suite passed **231 tests with zero failures and zero skips**. The Pump and Aalto route tests use real files. Security fixtures cover unregistered and corrupted weights, tampering after caching, explicit replacement, revoked registration, missing image models, and cache isolation across routes. Existing trusted sensor tests and unsupported-input tests also passed. Pyflakes and dependency checks passed.

All **39 real-file prediction checks** passed. The route report matched the pre-change report byte-for-byte, including predictions, displayed probabilities and checkpoint digests. A clean notebook and a real multipart Pump image upload also passed. No archived scores, checkpoints or scientific protocols changed.

Hosted CI runs the same security tests using temporary fixtures and skips the real-file tests when the local inputs are unavailable. See the [main workflow runs](https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment/actions?query=branch%3Amain) for the posted commit’s result.

**Phase 1 Complete With Scientific Limitations.** The application classifies current condition using laboratory analogues and synthetic imagery. Field generalization, calibrated failure probabilities, paired fusion and remaining useful life remain outside the verified scope.

## Closing checks · November 16, 2026

- **Image checkpoint trust on the Windows computer:** after the November 13 change, the seven local `.pt` files were compared with the digests in the committed `Results/deployment/route_check.json` before registration; all seven matched exactly. All 39 routes then passed.
- **Attempt G (planned in advance):** removing the Grader sensor model's absolute-level features made it independent of signal gain (86.7% at half and double amplitude, against 29.2% and 36.7%) and scored 86.7% on the fair test. Under added noise it scored 75.0%, below the archived model's 84.2% by more than the 5 points allowed, so it was not deployed. See [Grader attempt G](grader_gain_invariant_results.md).
- **Input requirements:** every sensor route rejects NaN and infinite values. No route can detect a wrong sampling rate or wrong units from the file, so uploads must use the publisher's rate and units.
- **Notebooks:** both notebooks execute without errors. The main notebook's improvement table now lists attempts A to E and G from the saved results.

The verdict is unchanged: **Phase 1 Complete With Scientific Limitations.**

## Final Phase 1 Cleanup and Verification · November 16, 2026

Baseline commit `7945df4a7cce2407dd6fbb2a2ce53ce9515730e2`. Verification ran in a separate checkout to preserve the local notebook edits.

- **Repository and dependencies.** Reviewed the 130 tracked files. Retained the project assets, research evidence and documented examples. The ignore rules now cover prediction history and allow the tracked methodology figure. Pyflakes and pip dependency checks passed. Existing requirements were retained.
- **Application and checkpoint trust.** The dashboard handles unreadable checkpoint-status metadata, rejects POSTs to unknown endpoints and neutralises spreadsheet formulas hidden behind whitespace in the history export. History hashes uploads using the existing streaming helper. The loaders still enforce registered checkpoint hashes and use weights-only loading for PyTorch.
- **Local inference.** All 39 real-file checks passed across eight equipment mappings, including the Motor Grader Aalto gear sensor, tooth image and original JUST bearing sensor. The local route report was byte-identical to the baseline. These checks establish deployment behavior. Generalization claims are based on the archived held-out experiments.
- **Notebooks and dashboard.** Both notebooks ran from clean kernels. The integrated notebook ran 3 executable cells and the research notebook ran 18. The dashboard accepted 17 real uploads covering eight sensor and eight image mappings plus Aalto, and the local HTTP and browser checks exercised history export, ambiguous input, search and status filtering.
- **Scientific verification.** Rechecked image split grouping and exact duplicates for the six available canonical image indexes. None had a shared recording unit or byte-identical image crossing a tested train/test boundary. That rerun did not cover the Pump dense-frame export or the Grader tooth source archive. The CPU benchmark now runs on macOS using peak RSS, with the Windows working-set measurement retained. The dataset check flagged eight missing archives among 163 checked entries and zero SHA-256 mismatches among files with an available reference digest. The missing archives are the MCC5-THU gearbox ZIP, MFGNet mesh ZIP and six JUST condition ZIPs. Full raw-data reproduction was therefore outside this run. Archived metrics, split definitions and checkpoint decisions, including the decision to keep Grader attempt G out of deployment, were preserved.
- **Automated verification.** The baseline suite passed 252 tests. The cleaned revision passed 255 with zero failures and zero skips locally. See the [Tests workflow](https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment/actions/workflows/tests.yml) for the hosted check. Hosted runners may skip real-file checks that require local fixtures.

**Phase 1 Complete With Scientific Limitations — Repository Cleaned and Verified.** The application classifies current component condition. Generalization to mining fleets, calibrated probabilities, paired sensor-image fusion and failure-time prediction remain unverified.

## Archived verification records

These records retain their original observations and decision criteria. Archived result files may cite their original filenames.

### Project audit and 20-day plan

November 5, 2026 · Group 9 · Baseline `05476ba`

The saved post-push check confirms eight sensor routes, eight image routes and 19 upload/error checks. The experimental results below are separate from those checks. The routes use deployment fits, including some trained on subsets different from the archived evaluations. A working route and its prediction confidence do not establish held-out accuracy.

This review uses the requirements recorded for the capstone discussion: compare sensor and visual methods for a comparable component fault, distinguish laboratory analogues from mine equipment, preserve the split, report class-level performance and explain the methodology without code snippets. The team previously stated a target above 80% for every equipment pair. Confirm the instructor's exact metric, split and treatment of analogues before treating that target as a formal rubric rule.

#### Eight equipment mappings

Scores are the archived primary held-out accuracy and macro F1, shown as percentages. The Grader image entry uses balanced accuracy. Test units are evaluation units, and several may share an underlying physical specimen.

| Equipment | Sensor accuracy / F1 | Image accuracy / F1 | Audit finding |
|---|---:|---:|---|
| Pump | 95.8 / 95.8, 24 records | 100 / 100, 24 records | Strongest aligned laboratory comparison. Both use healthy, inner race, outer race and rolling-element classes. The Ottawa and XJTU rigs are separate and use different held-out speed protocols. Verify specimen IDs before claiming generalization to new physical bearings. |
| Haul Truck | 80.0 / 74.1, 30 records | 58.3 / 56.6, 12 records | Four classes align on bearing analogues. UODS ball-fault recordings run unloaded while other classes run loaded, so load is confounded with ball fault. Sensor balanced accuracy is 73.3%. Image outer-race and ball recall are both 33.3%. Sensor holds out bearing IDs while vision holds out speed. |
| Belt Conveyor | 43.8 / 35.0, 16 runs | 50.0 / 43.2, 16 runs | Shared CORA gear-wear labels and run keys support a comparison at the experiment level. Sensor misses every wear25 run and images miss every wear75 run. The laboratory geared drive is the analogue. |
| Dozer | 43.8 / 35.0, 16 runs | 56.3 / 52.1, 16 runs | Reuses all 16 Belt Conveyor runs. Differences in visual scores do not establish a separate machine result. Report a second application mapping of the same benchmark. |
| Excavator | 25.0 / 18.3, 16 runs | 25.0 / 21.7, 16 runs | Same CORA classes across modalities, with both accuracies at the four-class baseline. Sensor bearing and misalignment recall are zero. Image healthy recall is zero. Local follow-up experiments use different targets or feature sets and need separate rows if presented. |
| Wheel Loader | 31.3 / 32.5, 16 runs | 18.8 / 20.6, 16 runs | Same CORA target across modalities. Images are below the four-class baseline and miss all broken-bar runs. Eight healthy or bearing run IDs also appear in Excavator. |
| Drill Rig | 73.3 / 70.5, 120 records | 88.9 / 88.8, 4,800 synthetic gears | The sensor study holds out speed and the image study holds out CAD designs. The image source calls R0 root breakage while the project maps it to the sensor class tooth crack. Treat that class as unaligned until justified. CAD results do not establish camera performance. Sensor healthy recall is 41.7%. [MCC5-THU data paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11070670/) · [MFGNet-Gear label definitions](https://huggingface.co/datasets/rsmei/MFGNet-Gear) |
| Motor Grader | 84.2 / 84.7, 120 records | 82.6 / 85.8, 2,879 tooth crops | Different components, labels, metrics and test units. JUST slewing-bearing tests hold out operating conditions on reused supports, one per class. The image result concerns good vs. damaged teeth and damaged recall is 66.1%. The earlier [gear-alignment feasibility review](motor_grader_gear_comparison.md) documents the missing independent physical gear IDs and full-gear label coverage. |

CORA's secondary same-recording time-split accuracy is 93.8%–100%, while the unseen-speed accuracy is 18.8%–56.3%. Each CORA row has four speeds and one run per class/speed. A single correct run changes pooled accuracy by 6.25 percentage points. Frame or window counts increase the number of observations within a run. They do not increase the number of independent test runs. The sensor selection for Belt Conveyor and Dozer also used vibration plus temperature in three of four folds, while their upload checkpoints use vibration alone. Keep those evaluation and deployment configurations distinct.

The README and methodology generalize that sensor and image data come from different rigs. CORA uses a shared rig and matching run IDs, with synchronization below the run level unverified. The methodology also describes an all-recording separation rule. That applies to the primary test, while the secondary time split deliberately reuses recordings. Correct both descriptions during implementation. [Repository results](../Results) · [Deployment verification](phase_1_enhancement_report.md#integrated-notebook-deployment-verification)

#### Ranked issues and acceptance criteria

| Rank | Severity and feasibility | Work after audit review | Acceptance criteria |
|---|---|---|---|
| 1 | Critical · High feasibility · 2 team-days | Correct the comparison matrix and professor-facing claims. Mark Grader as separate tasks, Dozer as a reused benchmark, Drill Rig as synthetic and partially aligned, and all routes as current-condition classification. | Every row names the component, class mapping, modality, rig, experimental unit, split, result provenance, and applicability. Published and demo results appear in separate tables. The team confirms whether >80% is strict and whether analogues count. |
| 2 | Critical · High feasibility for reporting, moderate for improvement · 4 team-days | Freeze CORA group manifests and prioritize one new benchmark experiment before expanding. Start with a binary healthy vs. worn gear comparison on the existing shared Belt Conveyor/Dozer runs, or a reproducible four-class baseline using the same unseen-speed folds. | A predeclared target and majority baseline. Zero cross-fold run leakage. One prediction per held-out run, fold-level results, class recall, macro F1 and balanced accuracy. Report binary and four-class scores separately. Stop after the planned comparison if performance remains weak. |
| 3 | High · Moderate feasibility · 3 team-days | Quantify Haul Truck load confounding and vision limits. Verify specimen metadata, report load by class, and use a load-matched common-label subset only if both modalities support it. | Whole-bearing grouping where IDs exist, with groups and classes counted before fitting. If ball fault remains inseparable from load, flag that class and report any narrower analysis separately. |
| 4 | High · High feasibility for scope correction · 2 team-days | Check Drill Rig fault mapping against the source, and trace Pump specimen grouping. | Source-backed label crosswalk. Drill Rig root breakage and root crack remain distinct unless the source evidence supports a common target. A three-class follow-up is allowed only with the same eligible classes on both sides and the original grouped split preserved. Pump claims remain limited to the observed split. |
| 5 | Critical mismatch · Low feasibility for replacement · 1 team-day timebox | Decide whether Grader can be aligned around gear teeth. Verify full-gear photographs, physical gear IDs, compatible damage labels, reuse rights and independent groups before training. | Proceed only with complete label and group evidence for both classes on both modalities. Otherwise keep the existing two experiments and state why their scores cannot be directly compared. |
| 6 | High demo impact · High feasibility · 2 team-days | Freeze the demo manifest and practice the technical explanation. | All 16 routes load from verified local checkpoint hashes. Confirm expected ambiguity and unsupported input behavior. Show file, mapping, prediction and confidence. Demonstrate shared-format limits and explain the data-to-result flow without code snippets. |

#### Schedule and decision gates

- **Days 1–3:** Review findings as a team, confirm the instructor's acceptance criteria, freeze the baseline commit, and produce the label, group and metric matrix.
- **Days 4–8:** Reconcile source IDs, archive the primary splits and make a timeboxed Grader data decision by Day 8. Prioritize the one CORA comparison with the clearest shared target.
- **Days 9–13:** If approved, run the prespecified CORA experiment and one bounded Haul Truck sensitivity analysis. Use separate output directories and keep the current deployment available. Record weak results without tuning on outer folds.
- **Days 14–16:** Review evidence and stop new modeling. Promote a candidate only if its comparable target and grouping are documented and its agreed primary score improves under the same evaluation protocol. Keep any remaining limits explicit.
- **Days 17–20:** Finalize the comparison table, methodology diagram, captions, technical narrative and demonstration. Reserve the final two days for an end-to-end run and corrections.

The audit itself made no changes to model code, weights, data or saved evaluations.

### Integrated notebook deployment verification

The November 5, 2026 post-push run passed all 16 inference routes and 19 upload/error checks. PASS means the route loaded its saved model and returned the equipment mapping, fault class, model confidence, and model name. Image samples include event-camera frames, infrared frames, and synthetic CAD renders.

| Equipment | Sensor CSV inference and sample prediction | Image inference | Automatic component-source identification |
|---|---|---|---|
| Pump | PASS · `I-C-1.csv` → `inner`, 99.9% | PASS · 4/4 sample labels matched | PASS · Supported source format |
| Haul Truck | PASS · `B_11_2.csv` → `ball`, 99.0% | PASS · 3/4 sample labels matched | PASS · Supported source format |
| Belt Conveyor | PASS · `W50_F50_S_Vx.csv` → `wear50`, 99.5% | PASS · 4/4 sample labels matched | NOT AVAILABLE · Shared CORA format |
| Dozer | PASS · `W50_F50_S_Vx.csv` → `wear50`, 99.5% | PASS · 4/4 sample labels matched | NOT AVAILABLE · Shared CORA format |
| Excavator | PASS · `BD_F50_S_Vx.csv` → `bearing`, 98.7% | PASS · 4/4 sample labels matched | NOT AVAILABLE · Shared CORA format |
| Wheel Loader | PASS · `HB_F50_S_Vx.csv` → `half_broken_bar`, 98.6% | PASS · 4/4 sample labels matched | NOT AVAILABLE · Shared CORA format |
| Drill Rig | PASS · `gear_pitting_H_torque_circulation_2000rpm_10Nm.csv` → `gear_pitting`, 92.9% | PASS · 2/4 sample labels matched | PASS · Supported source format |
| Motor Grader | PASS · `20221208-O-2rpm-0N-1.csv` → `outer_ring`, 79.6% | PASS · Separate gear-tooth model, selected manually | PASS · Supported sensor source format |

The Pump and Haul Truck CSVs preserve the samples and channels of their source MAT recordings. Their reported predictions matched the MAT routes. The four CORA mappings required explicit source confirmation. Exact local image fingerprints also resolved the Pump and Haul Truck event-camera exports and Drill Rig CAD renders. Filenames or a confirmed mapping describe a laboratory data source.

The notebook ran in a Jupyter kernel and exercised the upload widget callbacks with real CSV excerpts and image bytes. Checks covered unknown formats, invalid images, short recordings, incompatible mappings, shared CORA formats, and missing sensor and image checkpoints. The optional photo mode separately detected a Wheel Loader, Excavator and Dump Truck and returned ambiguity on a mixed scene. The current photo route treats all detector labels as unverified.

The Haul Truck CNN predicted outer-race fault for one ball-fault frame. The Drill Rig CNN predicted healthy for one wear render and one pitting render. The deployment checks included training samples. Confidence is a model output. For generalization, use the existing fair tests, which hold out recordings, speeds, bearings or designs. The existing time-split scores use later segments from recordings seen during training. CORA unseen-speed performance is weak in the saved results even when time-split scores are high.

Motor Grader slewing-bearing image diagnosis remains unavailable. The [JUST slewing-bearing dataset](https://data.mendeley.com/datasets/hwg8v5j8t6/1) supplies sensor signals. The separate [gear-tooth image dataset](https://universe.roboflow.com/vjs-jek7v/gearbox_adddingimg) labels a different component and task. Its deployment checkpoint predicts visible tooth condition from a manually confirmed crop. The Grader gear vibration model (Aalto, healthy vs tooth damage) has its own route since November 10; it does not replace the slewing-bearing route. All component routes classify current condition using laboratory analogues. Future failure and remaining useful life are outside the demonstrated scope.

The post-push record is `Processed_Data/pushed_deployment_verification.json`. Earlier route and photo checks are stored separately in `Processed_Data/final_deployment_verification.json` and `Processed_Data/whole_machine_verification.json`. The [evidence matrix](equipment_evidence_matrix.md) contains the archived held-out scores. Keep the trusted checkpoints and indexed image exports locally when running the demo.


#### November 8 regression check

The integrated notebook passed all 16 routes and 19 upload checks on real local samples after updating to `abd1d5d`. The repaired command-line verifier also passed all 16 routes across 38 prediction calls and checked 16 checkpoint hashes. The unsupported-file and unresolved-CORA checks passed. The verifier now finds the local Pump dense frame export, the Drill Rig and JUST deployment CSVs, and both Grader tooth crops. A route passes when it returns a prediction. Two Drill Rig image examples (wear and pitting) were predicted healthy in the CLI check. These are smoke tests, and their sample matches are not estimates of generalization.

The local rerun records are `Processed_Data/abd1d5d_notebook_verification.json` and `Processed_Data/nov8_cli_route_check.json`. The archived results were preserved. See the [Grader deployment decision](grader_matched_classes_results.md#deployment-decision--november-8-2026) for the Aalto checkpoint and data requirements.

#### November 10 check

After the security changes (`6a32997` and the November 10 commit) and the Aalto gear model, `route-check` passed all 39 routes: eight sensor routes, all image routes including both Grader tooth-photo crops, and the new Grader gear-sensor route (`GP1.csv` → healthy, 0.987). The Grader tooth-photo checkpoint was built from the November 2 Roboflow export with `--accept-archive`: its SHA-256 differs from the evaluated export, and it matched the evaluated crop counts and number of source groups (2,879 crops, 2,714 good and 165 damaged, 55 groups). The alternate export’s image content was not checked for exact equality. The unsupported-file and unresolved-CORA checks passed. The upload page returned the same Aalto prediction and refused requests from other websites, other host names, missing or negative lengths and non-form uploads. An untrusted model and a model changed by one byte were both refused. Four packages with known vulnerabilities (Pillow, filelock, tornado, certifi) were upgraded; pip-audit then found none. Both notebooks reran with 0 errors on the upgraded packages.

#### Grader image data and equipment photo follow-up

| Public source | What it supports | Deployment decision |
|---|---|---|
| [JUST slewing-bearing recordings](https://data.mendeley.com/datasets/hwg8v5j8t6/1) | Vibration and acoustic signals for healthy, inner ring, outer ring and rolling element conditions | Keep the existing sensor route. |
| [GearBox_adddingImg](https://universe.roboflow.com/vjs-jek7v/gearbox_adddingimg) | Visible gear-tooth damage | Keep the existing separate research result. These labels describe teeth, while the sensor target describes bearing parts. |
| [Bearing-ring surface inspection research](https://www.mdpi.com/1424-8220/25/21/6504) | Manufacturing surface defects on bearing rings | Useful background for inspection. Its manufacturing defect labels differ from the Grader slewing-bearing fault targets. |
| [ACID](https://www.acidb.net/dataset) and [AIDCON](https://www.ai2lab.org/aidcon/) | Equipment identity annotations, including Grader and Dozer. The full datasets require access requests. | Use the available ACID examples as a smoke test. A fair model comparison would require an independent annotated split. |

No reviewed image source established the Grader fault classes on independent bearing units. A fair Grader image benchmark would require images of the relevant component, verified labels, and multiple units per class. Hold out complete bearing units or machines so repeated views cannot cross the split. Compare a simple baseline with a transfer-learning classifier on identical held-out units using macro F1 and per-class recall.

The existing BuildVision photo detector was checked on four labeled ACID publisher examples. It proposed Excavator at 85.9% and Wheel Loader at 85.2% on their respective examples, returned Unknown Equipment on the Dozer, and incorrectly proposed Wheel Loader at 86.4% on the Grader. It also reported ambiguity on a mixed Excavator and Dump Truck scene from its publisher. These examples verify behavior and expose one failure. Dataset independence and generalization were not established. The notebook now returns Unknown Equipment for single photo detections and lists the suggestion, detector score, and pixel box separately. A mixed scene returns Ambiguous Equipment. Neither path invokes fault inference. The local record is `Processed_Data/photo_source_recheck.json`.


#### Inference routing regression checks

- **PASS** · Eight sensor and seven CNN routes ran through the Jupyter upload workflow. The 17 component upload checks covered unknown, ambiguous, invalid, short and missing-model inputs.
- **PASS** · The Grader photo that the detector scored as Wheel Loader returned Unknown Equipment with an unverified suggestion. The mixed photo returned Ambiguous Equipment. The photo mode emitted no fault classification.
- **PASS** · The command-line and upload server share the component routing in `predictive_maintenance.predict`. A shared CORA input without a mapping returned ambiguity. A confirmed CORA mapping ran its model. An arbitrary 224 × 224 image was rejected, while an indexed Drill Rig render ran its image model.
- **NOT AVAILABLE** · A Grader slewing-bearing image model still requires independently labeled visual inspections of relevant slewing bearings, including healthy, inner-ring, outer-ring and rolling-element conditions, with bearing IDs to keep units separate across training and testing. [JUST part 2](https://data.mendeley.com/datasets/rcxgmdxhbr/1) continues the sensor dataset. The [CITEF bearing benchmark](https://zenodo.org/records/8241764) also supplies vibration records.

The additional photo regression outputs are local at `Processed_Data/photo_routing_verification.json`.

The shared input parser was also verified directly through `predictive_maintenance.predict`. Pump and Haul Truck CSVs returned inner-race and ball faults respectively, matching the notebook. A CSV matching two different sensor schemas remained ambiguous even with a selected mapping. A blank image was rejected before scoring, including when Pump was selected.

The [Motor Grader image investigation](motor_grader_image_investigation.md) records source provenance, access and licensing evidence, the bearing-level data requirements, and the scope proposed for professor approval.


#### Motor Grader gear-tooth deployment verification

The version 1 COCO export dated November 5 contains 838 images. Its archive SHA-256 matches the local checkpoint metadata. The fixed ResNet18 feature and logistic regression deployment fit uses 2,879 labeled tooth crops from 55 source groups. The earlier 82.6% balanced accuracy belongs to the saved source-group-held-out experiment.

The integrated notebook upload callback, command line, and local multipart HTTP upload each returned `good_tooth` for one labeled good crop and `damaged_tooth` for one labeled damaged crop. These two training-sample checks establish inference behavior. Local outputs are saved in `Processed_Data/grader_gear_tooth_verification.json`. The Grader sensor checkpoint retained its SHA-256 throughout verification.

### Coding completion audit

AI-Based Predictive Maintenance for Mining Equipment · Group 9

**Verdict · Coding Complete.** All eight equipment mappings passed local sensor and image checks. The additional Motor Grader Aalto gear-vibration route passed with the publisher’s 250 RPM recording and a locally rebuilt checkpoint.

#### Completed features

- The [integrated notebook](../Notebook/Integrated_Equipment_Fault_Prediction.ipynb), command line and local upload page accept supported sensor recordings and component images. They select the appropriate checkpoint, run the saved preprocessing and display a condition, model name and class probability.
- The application checks checkpoint trust for local Joblib models and uses weights-only loading for the PyTorch image models. Shared CORA formats return ambiguity until the user confirms a compatible mapping.
- Invalid files, malformed MAT recordings, unknown schemas and missing checkpoints produce error messages. The local route checker records checkpoint hashes, reports missing samples and saves the current run outside the archived results.
- The [README](../README.md#quick-start) documents setup, upload formats, model registration and the route-check command.

#### Equipment mapping status

PASS means a real input reached the appropriate local checkpoint and returned a prediction. The label and probability below are from the CLI sample. The [evidence matrix](equipment_evidence_matrix.md) contains held-out performance.

| Equipment mapping | Sensor result | Image result |
|---|---|---|
| Pump | PASS · `inner`, 99.9% | PASS · indexed event-camera frames |
| Haul Truck | PASS · `ball`, 99.0% | PASS · indexed event-camera frames |
| Belt Conveyor | PASS · `wear50`, 99.5% | PASS · CORA infrared frames |
| Dozer | PASS · `wear50`, 99.5% | PASS · CORA infrared frames |
| Excavator | PASS · `bearing`, 98.7% | PASS · CORA infrared frames |
| Wheel Loader | PASS · `half_broken_bar`, 98.6% | PASS · CORA infrared frames |
| Drill Rig | PASS · `gear_pitting`, 92.9% | PASS · indexed CAD renders |
| Motor Grader | PASS · JUST bearing `outer_ring`, 79.6%. Aalto gear `healthy`, 98.7%. | PASS · both `good_tooth` and `damaged_tooth` crops returned their expected labels |

#### Tests executed

| Check | Observed result |
|---|---|
| `python -m pytest -q -rs` | 53 passed, 0 failed, 0 skipped, 12 warnings. The real Aalto test passed. The warnings concern the future removal of scikit-learn's SVC probability option. |
| `python -m pyflakes Notebook/predictive_maintenance.py tests` | Passed. |
| `python -m pip check` | No broken requirements. |
| `python Notebook/predictive_maintenance.py route-check` | 39 prediction checks passed across all routes, including Aalto. Unsupported-file and ambiguous-CORA checks passed. The command exited 0. Local results are in `Processed_Data/deployment/route_check.json`. |
| Clean-kernel integrated notebook | All cells executed. All 16 real-file routes and 19 upload checks passed. The eight sensor labels and displayed probabilities agreed with the CLI within rounding precision. The checks validated checkpoint labels and preprocessing metadata. A clean-kernel rerun also predicted `healthy` at 98.7% on the real Aalto recording. |
| Research notebook | All 18 code cells executed in a clean kernel. |
| Dependency checks | `pip check` reported no broken requirements. `pip-audit` reported no known vulnerabilities in the verification environment. |
| Local upload server | 16 real-file multipart uploads and seven error or ambiguity cases passed. Cases covered a corrupted image, unsupported type, unknown CSV schema, ambiguous CORA input and missing sensor or image models. The real Aalto multipart upload returned `healthy` at 98.7%. Automated HTTP tests also covered a corrupted MAT upload. |

#### Motor Grader integration

- **JUST slewing-bearing vibration** and **gear-tooth inspection** both passed real-file checks. Select the Grader mapping explicitly for a single-tooth photo crop.
- **Aalto gear vibration** passed the real-file test, CLI, clean-kernel notebook and local upload page. The 250 RPM recording produced `healthy` at 98.7% with the trusted logistic regression checkpoint.
- The Aalto and tooth-image data use different physical gears. Aalto labels a gear-pair run and the image route labels a tooth crop. See the [matched-class results](grader_matched_classes_results.md) for the laboratory evaluation and its limitations.

#### Reliability fixes

- Joblib loading verifies the same bytes it deserializes. The cache uses file identity and change time as well as size and modification time. A regression test replaces a cached checkpoint with a different file of identical size and modification time and confirms rejection.
- A malformed trust manifest produces a clear validation error. The notebook shows its status and opens the prediction interface.
- The loopback upload server rejects short request bodies and malformed multipart forms before prediction. The test server uses an operating-system-assigned port and accepts requests on that port.

#### Remaining limitations and GitHub Actions

- No blocking application bugs remained in the checks of the available routes. Two Drill Rig image samples labeled gear wear and gear pitting were predicted healthy. The demo probabilities are model outputs. Performance claims should use the held-out results.
- Dozer and Belt Conveyor reuse CORA evidence. The shared schema requires a confirmed laboratory mapping. The Drill Rig image route uses synthetic CAD renders. The deployment checks establish local inference on the available component samples.
- The [previous audit commit’s workflow](https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment/actions/runs/42923427532) passed. The [main workflow runs](https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment/actions?query=branch%3Amain) show CI status for the reliability fixes. CI skips real-file cases whose local fixtures are absent.

#### Aalto real-file recovery and rebuild

The full Aalto archive matched the publisher SHA-256 recorded in the [dataset manifest](../Dataset_Manifest/manifest.md). The exact recording was restored to `Raw_Data/aalto_gear/3012Hz/250RPM/0_12Nm/Healthy/GP1.csv`. The existing extractor generated 12,750 windows from 425 runs across eight gear pairs. The saved feature cache and the original recording produced the same feature values within numerical tolerance.

The existing `python Notebook/predictive_maintenance.py grader-gear train-final` command selected logistic regression with acceleration, shaft orders and torque, and registered the rebuilt checkpoint in the local trust manifest. The checkpoint at `Processed_Data/models/grader_gear_sensor.joblib` is a fresh fit on the full dataset. Its SHA-256 is:

```text
5f9580644da0cfa366cfae2905073b8ad3d95ef288c98449c5720073a36235fa
```

The exact recording is part of the fitted dataset, so this prediction verifies deployment. The archived held-out experiment remains the source for the generalization estimate. Keep the recording and the trusted checkpoint on the demonstration computer. The raw archive, feature cache, checkpoint and trust manifest are excluded from Git.

### Phase 1 final verification plan

AI-Based Predictive Maintenance for Mining Equipment · Group 9 · November 11, 2026 · committed before any of these results existed

This file fixes, in advance, what the final Phase 1 verification runs. It is a verification pass: archived results and deployed checkpoints are not changed, and nothing here is promoted to deployment. Outputs go to `Results/experiments/phase_1/final_verification.json` and `Figures/experiments/phase_1/`. The historical runner is available in the [recorded revision](https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment/blob/48d95c50ad615eb4caa7f779d471ec981ea23058/Notebook/phase1_final_verification.py).

#### 1. Reproduction and calibration of the archived sensor fair tests (all eight mappings)

- Rebuild every sensor feature table from the raw files and rerun the archived fair test exactly (same folds, same candidate models and feature sets, model chosen inside the training groups).
- **Reproduced** means the recording-level confusion matrix and the chosen (feature set, model) per round are identical to the archived `Results/<machine>/sensor.json`.
- Save each test recording's mean class probability. Report, per mapping: multiclass Brier score, top-label expected calibration error (5 equal-width bins), mean top-label confidence for correct and wrong predictions, and a Wilson 95% interval for accuracy. With 16 to 120 test recordings these are descriptive only; no calibration is fitted and deployed outputs are not changed.

#### 2. Leakage audit

- Sensor tables: each recording has one class and one held-out group; each round holds out exactly its declared groups; no recording is on both sides; every recording is tested once across rounds; identical feature rows shared between different recordings are counted.
- Images: for every exported image index, exact duplicate files (SHA-256) are checked across each round's train and test sides, and whole recordings or source units must not cross the split. Near-duplicates that are not byte-identical are not detected by this check.

#### 3. Sensor robustness beyond the Pump

For Haul Truck (UODS), Drill Rig (MCC5-THU) and Motor Grader (JUST): in each archived fair-test round, the archived chosen (feature set, model) is refitted on the training groups, and only the held-out recordings are disturbed. The disturbances are the Pump set, unchanged: Gaussian noise at 10% of each channel's RMS (seed 613), half and double amplitude, random removal of 5% of samples, the first half of the recording, and resampling to 90% of the rate while the extractor still assumes the original rate. Multi-channel recordings get the same removal mask on every channel. Report accuracy, balanced accuracy, macro F1 and the share of recordings whose class is unchanged from the clean prediction. The clean case must reproduce the archived score. These are synthetic disturbances on laboratory data, not a field robustness claim.

#### 4. Planned experiment E: CORA transient runs, unseen speed

- **Why:** the CORA stationary fair test has one run per class and speed, so it cannot tell whether the failure at an unseen speed comes from too few runs or from the speed change itself. The transient start-up runs (5 repetitions per class and speed, already downloaded for attempt C) have never been used for an unseen-speed test.
- **Test:** Belt Conveyor, Excavator and Wheel Loader classes (Dozer reuses the Belt Conveyor runs). Four rounds, each holding out one speed (F5, F15, F50, F60) with all five repetitions, so 20 test runs per round and 80 in total. Attempt C's transient features (orders normalised with the measured speed), the same five candidate models, chosen inside the training speeds. One prediction per run, with the majority control.
- **Runs once.** No change after seeing the result.
- **Reading:** clearly above the 25% control would mean more runs per class help at an unseen speed; near the control would mean the speed change itself is the barrier. Either way the result is reported only. It is not deployed, because the deployed routes take stationary recordings, and it does not replace the archived stationary fair test.

#### 5. Other checks

- Rerun the committed Phase 1 runner's CORA candidates, Pump robustness and deployment probes on this computer and compare with `Results/experiments/phase_1/verification.json`.
- Recompute the reported JUST diagnostic values from the original archive member; record sampling rates that can be measured from time columns.
- CPU inference benchmark of the Pump sensor, Pump image and Aalto routes on this Windows computer, one fresh process per route, with peak memory.
- New automated tests: saved confusion matrices agree with every reported metric; scalers sit inside the model pipelines; model selection never sees test rows; the image model cache keeps routes apart and stays bounded; analytic checks of skewness, kurtosis, crest factor, peak-to-peak, Welch band power and scale stability.

## Module rename regression verification · November 17, 2026

The starting main revision was `b9eed23d9c67f1dc249614a6869ca2e474ddc512`. The active module is `Notebook/predictive_maintenance.py`. A tracked-file search found one outdated command in a `.gitignore` comment; it now names the current script. Historical result JSON files retain their original provenance strings.

The first `demo-check` run found four missing sample paths. The route checker already supported indexed Pump frames and alternate deployment recordings for Drill Rig and JUST Grader. The demo now reuses that sample selection and keeps its fault-label assertions. Its automatic Pump expectation follows the existing source-identification behavior: a unique local image-index hash identifies Pump; an unindexed event image remains ambiguous. Two regression tests cover fallback selection and both identification outcomes.

Executed after the fix:

- `python -m pytest -q -rs` — 257 passed, zero failed, zero skipped locally, with 18 existing scikit-learn deprecation warnings.
- `python -m pyflakes Notebook tests` — passed.
- `python Notebook/predictive_maintenance.py route-check` — 39 successful prediction checks across eight sensor and eight image mappings plus the additional Aalto gear route; unsupported and ambiguous inputs also handled.
- `python Notebook/predictive_maintenance.py demo-check` — all 11 scripted prediction/error checks passed.
- CLI `--help`, a real Aalto `predict --no-history`, and startup of `serve` with a successful dashboard request passed.
- CLI `data-check` correctly exited 1 and reported eight missing research archives among 163 entries. Available referenced digests had zero mismatches. The missing archives are the MCC5-THU ZIP, MFGNet mesh ZIP and six JUST condition ZIPs; local inference samples and trusted models were available.
- Real HTTP dashboard checks passed for 17 uploads, ambiguous CORA input, unsupported and corrupted files, history CSV export and the upload-size guard. Automated tests also cover Host/Origin enforcement, escaping, checkpoint SHA-256 trust, PyTorch weights-only loading and cache invalidation.
- Headless Chrome executed drag/drop into the actual file input, filename feedback, history search and status filtering successfully.
- Both notebooks executed from clean kernels, with 3 integrated and 18 research code cells.

The hosted workflow for the starting rename commit completed successfully: [run 44209800850](https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment/actions/runs/44209800850). Its job logs recorded 253 passed and two skipped tests. Those two hosted skips require local Pump/Aalto recordings and checkpoints, which were exercised successfully in the local run. The workflow uses Python 3.11, installs CPU PyTorch and project requirements, and runs lint and tests on pushes and pull requests. The verification fix is checked by the subsequent [main workflow run](https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment/actions?query=branch%3Amain).

These checks verify application behavior. Archived held-out metrics remain the basis for scientific generalization claims.
