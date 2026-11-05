# Integrated notebook deployment verification

Verified locally on November 5, 2026. PASS means the route loaded its saved model and returned the equipment mapping, fault class, model confidence, and model name. Image samples include event-camera frames, infrared frames, and synthetic CAD renders.

| Equipment | Sensor CSV inference and sample prediction | CNN image inference | Automatic component-source identification |
|---|---|---|---|
| Pump | PASS · `I-C-1.csv` → `inner`, 99.9% | PASS · 4/4 sample labels matched | PASS · Supported source format |
| Haul Truck | PASS · `B_11_2.csv` → `ball`, 99.0% | PASS · 3/4 sample labels matched | PASS · Supported source format |
| Belt Conveyor | PASS · `W50_F50_S_Vx.csv` → `wear50`, 99.5% | PASS · 4/4 sample labels matched | NOT AVAILABLE · Shared CORA format |
| Dozer | PASS · `W50_F50_S_Vx.csv` → `wear50`, 99.5% | PASS · 4/4 sample labels matched | NOT AVAILABLE · Shared CORA format |
| Excavator | PASS · `BD_F50_S_Vx.csv` → `bearing`, 98.7% | PASS · 4/4 sample labels matched | NOT AVAILABLE · Shared CORA format |
| Wheel Loader | PASS · `HB_F50_S_Vx.csv` → `half_broken_bar`, 98.6% | PASS · 4/4 sample labels matched | NOT AVAILABLE · Shared CORA format |
| Drill Rig | PASS · `gear_pitting_H_torque_circulation_2000rpm_10Nm.csv` → `gear_pitting`, 92.9% | PASS · 2/4 sample labels matched | PASS · Supported source format |
| Motor Grader | PASS · `20221208-O-2rpm-0N-1.csv` → `outer_ring`, 79.6% | NOT AVAILABLE | PASS · Supported source format |

The Pump and Haul Truck CSVs preserve the samples and channels of their source MAT recordings. Their reported predictions matched the MAT routes. The four CORA mappings required explicit source confirmation. Exact local image fingerprints also resolved the Pump and Haul Truck event-camera exports and Drill Rig CAD renders. Filenames or a confirmed mapping describe a laboratory data source.

The notebook ran in a Jupyter kernel and exercised the upload widget callbacks with real CSV excerpts and image bytes. Checks covered unknown formats, invalid images, short recordings, incompatible mappings, shared CORA formats, and missing sensor and image checkpoints. The optional photo mode separately detected a Wheel Loader, Excavator and Dump Truck and returned ambiguity on a mixed scene. It returns visible equipment identity only.

The Haul Truck CNN predicted outer-race fault for one ball-fault frame. The Drill Rig CNN predicted healthy for one wear render and one pitting render. The deployment checks included training samples. Confidence is a model output. For generalization, use the existing fair tests, which hold out recordings, speeds, bearings or designs. The existing time-split scores use later segments from recordings seen during training. CORA unseen-speed performance is weak in the saved results even when time-split scores are high.

Motor Grader image fault inference remains unavailable. The [JUST slewing-bearing dataset](https://data.mendeley.com/datasets/hwg8v5j8t6/1) supplies sensor signals. The separate [gear-tooth image dataset](https://universe.roboflow.com/vjs-jek7v/gearbox_adddingimg) labels a different component and task. All component routes classify current condition using laboratory analogues. Future failure and remaining useful life are outside the demonstrated scope.

Local test outputs are stored in `Processed_Data/final_deployment_verification.json` and `Processed_Data/whole_machine_verification.json`. Keep the trusted checkpoints and indexed image exports locally when running the demo.
