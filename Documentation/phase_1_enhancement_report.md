# Phase 1 enhancement and verification report

AI-Based Predictive Maintenance for Mining Equipment · Ganesh Mishra, Lakshya Sharma and Sagar Patel

**Verdict · Ready for Phase 2 With Limitations.** The local system is ready for application development. The remaining limits concern held-out generalization, input provenance, and the availability of full research archives.

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
