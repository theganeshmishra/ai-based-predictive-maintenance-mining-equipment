# Coding completion audit

AI-Based Predictive Maintenance for Mining Equipment · Group 9

**Verdict · Coding Complete.** All eight equipment mappings passed local sensor and image checks. The additional Motor Grader Aalto gear-vibration route passed with the publisher’s 250 RPM recording and a locally rebuilt checkpoint.

## Completed features

- The [integrated notebook](../Notebook/Integrated_Equipment_Fault_Prediction.ipynb), command line and local upload page accept supported sensor recordings and component images. They select the appropriate checkpoint, run the saved preprocessing and display a condition, model name and class probability.
- The application checks checkpoint trust for local Joblib models and uses weights-only loading for the PyTorch image models. Shared CORA formats return ambiguity until the user confirms a compatible mapping.
- Invalid files, malformed MAT recordings, unknown schemas and missing checkpoints produce error messages. The local route checker records checkpoint hashes, reports missing samples and saves the current run outside the archived results.
- The [README](../README.md#quick-start) documents setup, upload formats, model registration and the route-check command.

## Equipment mapping status

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

## Tests executed

| Check | Observed result |
|---|---|
| `python -m pytest -q -rs` | 53 passed, 0 failed, 0 skipped, 12 warnings. The real Aalto test passed. The warnings concern the future removal of scikit-learn's SVC probability option. |
| `python -m pyflakes Notebook/pdm.py tests` | Passed. |
| `python -m pip check` | No broken requirements. |
| `python Notebook/pdm.py route-check` | 39 prediction checks passed across all routes, including Aalto. Unsupported-file and ambiguous-CORA checks passed. The command exited 0. Local results are in `Processed_Data/deployment/route_check.json`. |
| Clean-kernel integrated notebook | All cells executed. All 16 real-file routes and 19 upload checks passed. The eight sensor labels and displayed probabilities agreed with the CLI within rounding precision. The checks validated checkpoint labels and preprocessing metadata. A clean-kernel rerun also predicted `healthy` at 98.7% on the real Aalto recording. |
| Research notebook | All 18 code cells executed in a clean kernel. |
| Dependency checks | `pip check` reported no broken requirements. `pip-audit` reported no known vulnerabilities in the verification environment. |
| Local upload server | 16 real-file multipart uploads and seven error or ambiguity cases passed. Cases covered a corrupted image, unsupported type, unknown CSV schema, ambiguous CORA input and missing sensor or image models. The real Aalto multipart upload returned `healthy` at 98.7%. Automated HTTP tests also covered a corrupted MAT upload. |

## Motor Grader integration

- **JUST slewing-bearing vibration** and **gear-tooth inspection** both passed real-file checks. Select the Grader mapping explicitly for a single-tooth photo crop.
- **Aalto gear vibration** passed the real-file test, CLI, clean-kernel notebook and local upload page. The 250 RPM recording produced `healthy` at 98.7% with the trusted logistic regression checkpoint.
- The Aalto and tooth-image data use different physical gears. Aalto labels a gear-pair run and the image route labels a tooth crop. See the [matched-class results](grader_matched_classes_results.md) for the laboratory evaluation and its limitations.

## Reliability fixes

- Joblib loading verifies the same bytes it deserializes. The cache uses file identity and change time as well as size and modification time. A regression test replaces a cached checkpoint with a different file of identical size and modification time and confirms rejection.
- A malformed trust manifest produces a clear validation error. The notebook shows its status and opens the prediction interface.
- The loopback upload server rejects short request bodies and malformed multipart forms before prediction. The test server uses an operating-system-assigned port and accepts requests on that port.

## Remaining limitations and GitHub Actions

- No blocking application bugs remained in the checks of the available routes. Two Drill Rig image samples labeled gear wear and gear pitting were predicted healthy. The demo probabilities are model outputs. Performance claims should use the held-out results.
- Dozer and Belt Conveyor reuse CORA evidence. The shared schema requires a confirmed laboratory mapping. The Drill Rig image route uses synthetic CAD renders. The deployment checks establish local inference on the available component samples.
- The [previous audit commit’s workflow](https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment/actions/runs/42923427532) passed. The [main workflow runs](https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment/actions?query=branch%3Amain) show CI status for the reliability fixes. CI skips real-file cases whose local fixtures are absent.

## Aalto real-file recovery and rebuild

The full Aalto archive matched the publisher SHA-256 recorded in the [dataset manifest](../Dataset_Manifest/manifest.md). The exact recording was restored to `Raw_Data/aalto_gear/3012Hz/250RPM/0_12Nm/Healthy/GP1.csv`. The existing extractor generated 12,750 windows from 425 runs across eight gear pairs. The saved feature cache and the original recording produced the same feature values within numerical tolerance.

The existing `python Notebook/pdm.py grader-gear train-final` command selected logistic regression with acceleration, shaft orders and torque, and registered the rebuilt checkpoint in the local trust manifest. The checkpoint at `Processed_Data/models/grader_gear_sensor.joblib` is a fresh fit on the full dataset. Its SHA-256 is:

```text
5f9580644da0cfa366cfae2905073b8ad3d95ef288c98449c5720073a36235fa
```

The exact recording is part of the fitted dataset, so this prediction verifies deployment. The archived held-out experiment remains the source for the generalization estimate. Keep the recording and the trusted checkpoint on the demonstration computer. The raw archive, feature cache, checkpoint and trust manifest are excluded from Git.
