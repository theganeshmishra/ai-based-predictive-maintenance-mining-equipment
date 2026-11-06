# Integrated notebook deployment verification

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

Motor Grader slewing-bearing image diagnosis remains unavailable. The [JUST slewing-bearing dataset](https://data.mendeley.com/datasets/hwg8v5j8t6/1) supplies sensor signals. The separate [gear-tooth image dataset](https://universe.roboflow.com/vjs-jek7v/gearbox_adddingimg) labels a different component and task. Its deployment checkpoint was built after this verification and smoke-tested on a labeled tooth crop; this does not add a slewing-bearing image route. All component routes classify current condition using laboratory analogues. Future failure and remaining useful life are outside the demonstrated scope.

The post-push record is `Processed_Data/pushed_deployment_verification.json`. Earlier route and photo checks are stored separately in `Processed_Data/final_deployment_verification.json` and `Processed_Data/whole_machine_verification.json`. The [evidence matrix](equipment_evidence_matrix.md) contains the archived held-out scores. Keep the trusted checkpoints and indexed image exports locally when running the demo.


## Grader image data and equipment photo follow-up

| Public source | What it supports | Deployment decision |
|---|---|---|
| [JUST slewing-bearing recordings](https://data.mendeley.com/datasets/hwg8v5j8t6/1) | Vibration and acoustic signals for healthy, inner ring, outer ring and rolling element conditions | Keep the existing sensor route. |
| [GearBox_adddingImg](https://universe.roboflow.com/vjs-jek7v/gearbox_adddingimg) | Visible gear-tooth damage | Keep the existing separate research result. These labels describe teeth, while the sensor target describes bearing parts. |
| [Bearing-ring surface inspection research](https://www.mdpi.com/1424-8220/25/21/6504) | Manufacturing surface defects on bearing rings | Useful background for inspection. Its manufacturing defect labels differ from the Grader slewing-bearing fault targets. |
| [ACID](https://www.acidb.net/dataset) and [AIDCON](https://www.ai2lab.org/aidcon/) | Equipment identity annotations, including Grader and Dozer. The full datasets require access requests. | Use the available ACID examples as a smoke test. A fair model comparison would require an independent annotated split. |

No reviewed image source established the Grader fault classes on independent bearing units. A fair Grader image benchmark would require images of the relevant component, verified labels, and multiple units per class. Hold out complete bearing units or machines so repeated views cannot cross the split. Compare a simple baseline with a transfer-learning classifier on identical held-out units using macro F1 and per-class recall.

The existing BuildVision photo detector was checked on four labeled ACID publisher examples. It proposed Excavator at 85.9% and Wheel Loader at 85.2% on their respective examples, returned Unknown Equipment on the Dozer, and incorrectly proposed Wheel Loader at 86.4% on the Grader. It also reported ambiguity on a mixed Excavator and Dump Truck scene from its publisher. These examples verify behavior and expose one failure. Dataset independence and generalization were not established. The notebook now returns Unknown Equipment for single photo detections and lists the suggestion, detector score, and pixel box separately. A mixed scene returns Ambiguous Equipment. Neither path invokes fault inference. The local record is `Processed_Data/photo_source_recheck.json`.


## Inference routing regression checks

- **PASS** · Eight sensor and seven CNN routes ran through the Jupyter upload workflow. The 17 component upload checks covered unknown, ambiguous, invalid, short and missing-model inputs.
- **PASS** · The Grader photo that the detector scored as Wheel Loader returned Unknown Equipment with an unverified suggestion. The mixed photo returned Ambiguous Equipment. The photo mode emitted no fault classification.
- **PASS** · The command-line and upload server share the component routing in `pdm.predict`. A shared CORA input without a mapping returned ambiguity. A confirmed CORA mapping ran its model. An arbitrary 224 × 224 image was rejected, while an indexed Drill Rig render ran its image model.
- **NOT AVAILABLE** · A Grader slewing-bearing image model still requires independently labeled visual inspections of relevant slewing bearings, including healthy, inner-ring, outer-ring and rolling-element conditions, with bearing IDs to keep units separate across training and testing. [JUST part 2](https://data.mendeley.com/datasets/rcxgmdxhbr/1) continues the sensor dataset. The [CITEF bearing benchmark](https://zenodo.org/records/8241764) also supplies vibration records.

The additional photo regression outputs are local at `Processed_Data/photo_routing_verification.json`.

The shared input parser was also verified directly through `pdm.predict`. Pump and Haul Truck CSVs returned inner-race and ball faults respectively, matching the notebook. A CSV matching two different sensor schemas remained ambiguous even with a selected mapping. A blank image was rejected before scoring, including when Pump was selected.

The [Motor Grader image investigation](motor_grader_image_investigation.md) records source provenance, access and licensing evidence, the bearing-level data requirements, and the scope proposed for professor approval.


## Motor Grader gear-tooth deployment verification

The version 1 COCO export dated November 5 contains 838 images. Its archive SHA-256 matches the local checkpoint metadata. The fixed ResNet18 feature and logistic regression deployment fit uses 2,879 labeled tooth crops from 55 source groups. The earlier 82.6% balanced accuracy belongs to the saved source-group-held-out experiment.

The integrated notebook upload callback, command line, and local multipart HTTP upload each returned `good_tooth` for one labeled good crop and `damaged_tooth` for one labeled damaged crop. These two training-sample checks establish inference behavior. Local outputs are saved in `Processed_Data/grader_gear_tooth_verification.json`. The Grader sensor checkpoint retained its SHA-256 throughout verification.
