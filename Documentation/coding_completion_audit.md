# PROJ 613 coding completion audit

AI-Based Predictive Maintenance for Mining Equipment · Group 9

**Verdict · Complete With Limitations.** All eight equipment mappings have working sensor and image inference on the verification computer. The additional Motor Grader Aalto gear-vibration route still requires a real-file test on this computer.

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
| Motor Grader | PASS · JUST bearing `outer_ring`, 79.6% | PASS · both `good_tooth` and `damaged_tooth` crops returned their expected labels |

## Tests executed

| Check | Observed result |
|---|---|
| `python -m pytest -q -rs` | 52 passed, 1 skipped, 12 warnings. The skip is the real Aalto prediction test. The warnings concern the future removal of scikit-learn's SVC probability option. |
| `python -m pyflakes Notebook/pdm.py tests` | Passed. |
| `python -m pip check` | No broken requirements. |
| `python Notebook/pdm.py route-check` | 38 prediction calls returned results across the 16 available mappings. Unsupported-file and ambiguous-CORA checks passed. The command exited 2 because the additional Aalto sample was absent. Local results are in `Processed_Data/deployment/route_check.json`. |
| Clean-kernel integrated notebook | All cells executed. All 16 real-file routes and 19 upload checks passed. The eight sensor labels and displayed probabilities agreed with the CLI within rounding precision. The checks validated checkpoint labels and preprocessing metadata. A separate synthetic Aalto upload passed. |
| Research notebook | All 18 code cells executed in a clean kernel. |
| Dependency checks | `pip check` reported no broken requirements. `pip-audit` reported no known vulnerabilities in the verification environment. |
| Local upload server | 16 real-file multipart uploads and seven error or ambiguity cases passed. Cases covered a corrupted image, unsupported type, unknown CSV schema, ambiguous CORA input and missing sensor or image models. Automated HTTP tests also covered a corrupted MAT upload. |

## Motor Grader integration

- **JUST slewing-bearing vibration** and **gear-tooth inspection** both passed real-file checks. Select the Grader mapping explicitly for a single-tooth photo crop.
- **Aalto gear vibration** selected the dedicated gear checkpoint and returned the binary condition with a synthetic test estimator. The trained-model test is skipped on this checkout. It needs `Raw_Data/aalto_gear/3012Hz/250RPM/0_12Nm/Healthy/GP1.csv` and `Processed_Data/models/grader_gear_sensor.joblib`. The local trust manifest exists. A verified copy of the fitted checkpoint must be registered locally before use.
- The Aalto and tooth-image data use different physical gears. Aalto labels a gear-pair run and the image route labels a tooth crop. See the [matched-class results](grader_matched_classes_results.md) for the laboratory evaluation and its limitations.

## Reliability fixes

- Joblib loading verifies the same bytes it deserializes. The cache uses file identity and change time as well as size and modification time. A regression test replaces a cached checkpoint with a different file of identical size and modification time and confirms rejection.
- A malformed trust manifest produces a clear validation error. The notebook shows its status and opens the prediction interface.
- The loopback upload server rejects short request bodies and malformed multipart forms before prediction. The test server uses an operating-system-assigned port and accepts requests on that port.

## Remaining limitations and GitHub Actions

- No blocking application bugs remained in the checks of the available routes. Two Drill Rig image samples labeled gear wear and gear pitting were predicted healthy. The demo probabilities are model outputs. Performance claims should use the held-out results.
- Dozer and Belt Conveyor reuse CORA evidence. The shared schema requires a confirmed laboratory mapping. The Drill Rig image route uses synthetic CAD renders. The deployment checks establish local inference on the available component samples.
- The [previous audit commit’s workflow](https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment/actions/runs/42923427532) passed. The [main workflow runs](https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment/actions?query=branch%3Amain) show CI status for the reliability fixes. CI skips real-file cases whose local fixtures are absent.

## Aalto real-file recovery check

A fresh local run completed with **52 passed, 0 failed, and 1 skipped**. The CLI returned 38 predictions and reported one missing Aalto sample with exit code 2. The real Aalto test remains skipped because the exact CSV and fitted checkpoint listed above are absent. The searched local project and shared folders, accessible Drive, GitHub releases and Actions artifacts contained neither file.

The [original deployment record](../Results/deployment/route_check.json) records a healthy prediction of 0.987 for `GP1.csv` and the original `grader_gear_sensor.joblib` SHA-256:

```text
418fb49af15362f033004109561745ce14a736f7b798daeb85a5756b1d325138
```

Recover the original checkpoint and sample from the environment that produced that record. Verify the source and match the recorded checkpoint hash before registering it in the local trust manifest. The publisher also lists and serves the 9,256,917,062-byte source archive with the SHA-256 in the [dataset manifest](../Dataset_Manifest/manifest.md). This archive provides the recordings for an independently rebuilt checkpoint through the existing `grader-gear train-final` command. No archive or fitted model was recovered in this check. [Aalto Gear Failure Dataset 3012 Hz v2](https://data.mendeley.com/datasets/fywnj597d8/2)

After restoring both files, run `python -m pytest -q tests/test_features_and_routes.py::test_grader_gear_route`, then rerun the full suite and `python Notebook/pdm.py route-check`. The prior deployment result documents the original environment. This checkout requires a fresh real-file result before the Aalto route can be signed off.

The [workflow for the tested software revision](https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment/actions/runs/42924394880) passed. The [main workflow runs](https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment/actions?query=branch%3Amain) show CI status for this audit update.
