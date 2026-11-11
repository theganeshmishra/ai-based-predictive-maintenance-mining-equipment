# Phase 1 enhancement and verification report

AI-Based Predictive Maintenance for Mining Equipment · Ganesh Mishra, Lakshya Sharma and Sagar Patel

**Final verdict (November 11) · Phase 1 Complete With Scientific Limitations.** See the final verification section at the end. **Earlier verdict · Ready for Phase 2 With Limitations.** The local system is ready for application development. The remaining limits concern held-out generalization, input provenance, and the availability of full research archives.

## Baseline and changes

The work started from `044d428ab85bf46631fad34f5f5c2722481d42f7` on GitHub main. The baseline rerun had 53 passed tests, zero failures or skips, and 39 successful prediction checks. Pyflakes passed. The existing application already supported eight mappings, three prediction interfaces, separate Grader bearing and gear tasks, and checkpoint trust verification. [Coding audit](coding_completion_audit.md) · [Scientific validation](scientific_validation.md)

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

From the repository root with the project environment active:

```bash
python Notebook/phase1_validation.py
python Notebook/phase1_validation.py --benchmark pump_image
python -m pytest -q -rs
python -m pyflakes Notebook/pdm.py tests
python -m pyflakes Notebook tests
python Notebook/pdm.py route-check
```

The experiment runner defaults to `Processed_Data/phase1/` for outputs. It needs the Ottawa recordings, CORA stationary Vx recordings, the JUST demonstration excerpt, trusted local checkpoints, indexed image exports and the Aalto demo CSV. The paired benchmark uses an isolated checkout of `044d428` with access to the same local data and the new benchmark runner. Raw files and fitted models remain local.

## Regression and remaining limitations

- Local verification returned **69 passed, 0 failed, 0 skipped**. The real Pump and Grader Aalto tests ran alongside synthetic and mocked tests. Pyflakes and `pip check` passed. Existing scikit-learn SVC deprecation warnings remain.
- **39 real-file prediction checks passed**. The CLI also rejected an unsupported format and returned ambiguity for an unconfirmed CORA file. The integrated notebook passed 19 upload checks and a separate real Aalto check. The local web server passed 16 real uploads, six failure or ambiguity checks, and an additional real Aalto multipart upload.
- `data-check` listed eight missing research inputs. These are the MCC5-THU archive, MFGNet mesh archive and six JUST condition archives. Local deployment files cover every prediction route. The application does not require the full research archives for the demo.
- Tracked paths and documentation were rechecked for obsolete course naming and assistant artifacts. The existing ignore rules cover the local assistant files. [GitHub Actions for main](https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment/actions?query=branch%3Amain) runs the expanded lint and test suite. Hosted CI skips real-file tests when the local fixtures are absent.

## Final Phase 1 verification (November 11, 2026)

**Verdict · Phase 1 Complete With Scientific Limitations.** Every archived sensor result reproduces exactly from the raw data, no leakage was found, all tests and prediction routes pass, and two deployment defects found during the checks were fixed. The limits are scientific: small held-out sets, laboratory analogues only, four CORA mappings that do not generalise to an unseen speed, uncalibrated probabilities, and strong sensitivity of some sensor models to gain, noise and sampling-rate errors.

The checks and the one new experiment were fixed in advance in the [final verification plan](phase_1_final_verification_plan.md), committed before they ran. Runner: `python Notebook/phase1_final_verification.py`. Results: [final_verification.json](../Results/experiments/phase_1/final_verification.json). Archived results and deployed checkpoints are unchanged.

### 1. Commits and environment

- Start: `481d2e5e9b94027fae4a0cba1e575f925511067c`, GitHub main after a history rewrite pushed from another computer. This computer's earlier commits had identical content and were moved onto it. The final commit is the one that adds this section.
- Verified on Windows 11 (build 26200), Python 3.11.9, NumPy 2.4.6, pandas 2.3.3, scikit-learn 1.9.1, PyTorch 2.14.1+cpu, seed 613. The earlier Phase 1 figures were measured on Apple Silicon macOS, Python 3.12.14.

### 2. Tests run in this verification

| Check | Result |
|---|---|
| `python -m pytest -q -rs` at the start (481d2e5) | 69 passed, 0 failed, 0 skipped |
| Same, final (6 consecutive runs) | **230 passed, 0 failed, 0 skipped** in each run |
| `python -m pyflakes Notebook tests` | clean |
| `python Notebook/pdm.py route-check`, final | **39 of 39 routes passed**, 0 failed; unsupported file rejected; unmapped CORA file returned "Ambiguous Equipment" |
| `pip check` | no broken requirements |

The 161 new test cases are in `tests/test_phase1_final.py`. 148 of them check that every saved confusion matrix in `Results/` agrees with its reported accuracy, balanced accuracy, macro F1 and unit count. The rest are leakage guards, cache bounds, analytic diagnostics checks and UODS trimming. They are synthetic or read committed result files, so they also run on GitHub. The real-file tests (Pump and Aalto routes) ran locally and are skipped on GitHub, where the data is absent.

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
| Partially reproducible | the full `phase1_validation.py` run: its diagnostic figure reads `Processed_Data/deployment_samples/grader/20221208-O-2rpm-0N-1.csv`, which exists only on the computer that produced it; the same file is inside `condition1.zip` |
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
- **Gap:** `.pt` image checkpoints are loaded with `weights_only=True`, which prevents code execution. But they aren't on the SHA-256 trust list, so a replaced image checkpoint would be used without warning. Adding them to the trust list is recommended before any shared deployment.

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
