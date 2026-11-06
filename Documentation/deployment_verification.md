# Integrated notebook deployment verification

> On 2026-11-06 the Grader was moved off `main` to branch `grader/full-pipeline`, together with its rows and sections of this record. Counts below are as recorded on November 5, when the Grader was still included.

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

The Pump and Haul Truck CSVs preserve the samples and channels of their source MAT recordings. Their reported predictions matched the MAT routes. The four CORA mappings required explicit source confirmation. Exact local image fingerprints also resolved the Pump and Haul Truck event-camera exports and Drill Rig CAD renders. Filenames or a confirmed mapping describe a laboratory data source.

The notebook ran in a Jupyter kernel and exercised the upload widget callbacks with real CSV excerpts and image bytes. Checks covered unknown formats, invalid images, short recordings, incompatible mappings, shared CORA formats, and missing sensor and image checkpoints. The optional photo mode separately detected a Wheel Loader, Excavator and Dump Truck and returned ambiguity on a mixed scene. The current photo route treats all detector labels as unverified.

The Haul Truck CNN predicted outer-race fault for one ball-fault frame. The Drill Rig CNN predicted healthy for one wear render and one pitting render. The deployment checks included training samples. Confidence is a model output. For generalization, use the existing fair tests, which hold out recordings, speeds, bearings or designs. The existing time-split scores use later segments from recordings seen during training. CORA unseen-speed performance is weak in the saved results even when time-split scores are high.

The post-push record is `Processed_Data/pushed_deployment_verification.json`. Earlier route and photo checks are stored separately in `Processed_Data/final_deployment_verification.json` and `Processed_Data/whole_machine_verification.json`. The [evidence matrix](equipment_evidence_matrix.md) contains the archived held-out scores. Keep the trusted checkpoints and indexed image exports locally when running the demo.


## Equipment photo follow-up

The existing BuildVision photo detector was checked on four labeled ACID publisher examples. It proposed Excavator at 85.9% and Wheel Loader at 85.2% on their respective examples, returned Unknown Equipment on the Dozer, and incorrectly proposed Wheel Loader at 86.4% on the Grader. It also reported ambiguity on a mixed Excavator and Dump Truck scene from its publisher. These examples verify behavior and expose one failure. Dataset independence and generalization were not established. The notebook now returns Unknown Equipment for single photo detections and lists the suggestion, detector score, and pixel box separately. A mixed scene returns Ambiguous Equipment. Neither path invokes fault inference. The local record is `Processed_Data/photo_source_recheck.json`.


## Inference routing regression checks

- **PASS** · Eight sensor and seven CNN routes ran through the Jupyter upload workflow. The 17 component upload checks covered unknown, ambiguous, invalid, short and missing-model inputs.
- **PASS** · The command-line and upload server share the component routing in `pdm.predict`. A shared CORA input without a mapping returned ambiguity. A confirmed CORA mapping ran its model. An arbitrary 224 × 224 image was rejected, while an indexed Drill Rig render ran its image model.

The additional photo regression outputs are local at `Processed_Data/photo_routing_verification.json`.

The shared input parser was also verified directly through `pdm.predict`. Pump and Haul Truck CSVs returned inner-race and ball faults respectively, matching the notebook. A CSV matching two different sensor schemas remained ambiguous even with a selected mapping. A blank image was rejected before scoring, including when Pump was selected.
