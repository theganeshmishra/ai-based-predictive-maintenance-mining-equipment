# Public dataset shortlist for mining-equipment maintenance

**Status: unverified candidates.** This list was compiled on 2026-10-02 from an
AI assistant chat session. Dataset descriptions, sizes, licences and download
availability have **not** been independently checked, and the original links
were not preserved. Find each dataset by name and confirm its licence, content
and availability at the source before relying on it. Nothing in this file is
TiNA data, and it does not change the project scope (TiNA sensor data is the
primary module; any vision work stays a separate module).

## What these are, and are not

- Most entries are **component-level experimental datasets** (bearings,
  gearboxes, hydraulic rigs). They are not recordings from mining machines.
  Any mapping to excavators, crushers, conveyors and so on is a *potential
  application*, not something the data demonstrates.
- No single public dataset in this list provides **matched images, sensor
  readings and failure histories across a mining fleet**. Do not combine
  unrelated records as if they came from the same machine.
- Images of existing defects support **inspection**. They do not establish
  remaining useful life on their own.

## Dataset catalog

| Dataset | Data type | Coverage and purpose | Main limitation |
|---|---|---|---|
| UCI Hydraulic Systems | Sensor time series | Pump, valve, cooler and accumulator conditions; 2,205 test cycles | Hydraulic test rig; condition diagnosis rather than failure-time prediction |
| NASA IMS Bearings | Vibration | Bearing degradation experiments; useful for developing failure prediction | Experimental bearings, not mining fleet recordings |
| Paderborn Bearing Dataset | Vibration and motor current | Healthy and damaged bearings under different operating conditions | Fault diagnosis; commercial use requires contacting the provider |
| Case Western Reserve (CWRU) Bearing Dataset | Vibration | Normal bearings and inner-race, outer-race and rolling-element defects | Artificially introduced faults; not run-to-failure histories |
| PHM 2009 Gearbox | Vibration and tachometer | Gearbox fault type, location and severity; a labelled version is listed | Limited fault progression; older download links may need checking |
| NASA C-MAPSS | Simulated sensor time series | Engine degradation and remaining-useful-life modelling | Simulated aircraft turbofans, not mining diesel engines |
| MIMII | Audio | Normal and abnormal sounds from pumps, valves, fans and slide rails | Industrial anomaly detection; no failure-time labels |
| BeltCrack | Sequential images | Industrial conveyor belt crack detection | Covers conveyors only; crack detection does not by itself predict time to failure |
| Iron Ore Conveyor Anomalies | Images and video | Normal and abnormal subsets; foreign objects on a conveyor; some YOLO annotations; 5.8 GB; reported as CC BY 4.0 | Laboratory conveyor; annotations are partial |
| MVTec AD 2 | Images and defect masks | More than 8,000 industrial inspection images across eight scenarios | General visual inspection, not mining equipment; noncommercial licence |
| AI4I 2020 Predictive Maintenance | Synthetic tabular data | 10,000 records for practising machine-failure classification | Synthetic, generic machinery data |
| Mining Haul Truck Maintenance Demo | Synthetic sensor and maintenance tables | Haul truck fleet, sensor readings, failures and maintenance records | Portfolio demonstration using generated data, not real mine records |

## Possible equipment mapping (applications, not evidence)

| Equipment | Datasets that might inform a prototype |
|---|---|
| Excavators, loaders, hydraulic drills | UCI Hydraulic Systems |
| Crushers, mills, pumps, motor drives | Paderborn Bearing Dataset |
| Gearboxes in crushers, conveyors and drivetrains | PHM 2009 Gearbox |
| Rotating equipment (bearing degradation) | NASA IMS Bearings |
| Conveyor camera inspection | BeltCrack, Iron Ore Conveyor Anomalies |

## Suggested starting points (from the same chat)

- Multi-equipment prototype with sensors: UCI Hydraulic Systems, NASA IMS
  Bearings, PHM 2009 Gearbox and MIMII together cover hydraulic, bearing,
  gearbox and acoustic monitoring. They are experimental data, so any model
  still needs validation on real mining equipment data.
- Failure prediction from degradation: NASA IMS Bearings.
- Camera-based conveyor inspection with OpenCV: BeltCrack.
- Hydraulic condition monitoring: UCI Hydraulic Systems.

## What a maintenance-timing model needs

To predict *when* maintenance will be needed, public datasets are not enough.
The target system needs timestamped observations linked to machine IDs,
operating hours and load, and maintenance or failure records. For the visual
side, collect inspection images of leaks, cracks, tyre damage, corrosion and
worn teeth, tagged with equipment ID, date and defect type, then link them to
sensor readings and maintenance records.

## Fit with this project

- The proposal names TiNA (primary, sensor) and the KIT ball-screw-drive
  surface-defect set (vision). Neither is part of this list.
- Sensor and image sources come from different equipment, so they can only be
  linked at the decision level, as the proposal already states.
- Image sources the team has individually checked belong in the OpenCV track
  (`notebooks/OpenCV_Mining_Equipment_Damage.ipynb`). Treat entries here as
  leads until someone has verified them.
