# Dataset manifest

The project uses public laboratory benchmarks, synthetic CAD renders, and small-gear inspection photos as equipment analogues. The [evidence matrix](../Documentation/equipment_evidence_matrix.md) records the component and label alignment of each mapping. Files are downloaded once by hand into `Raw_Data/` (git-ignored, never committed). `python Notebook/pdm.py data-check` lists anything missing and checks the SHA-256 where it is known.

| Machine | Side | Dataset | Source | Licence | Local folder |
|---|---|---|---|---|---|
| Pump | sensor | University of Ottawa bearing rig, vibration (Huang and Baddour) | [Mendeley Data v43hmbwxpm v2](https://data.mendeley.com/datasets/v43hmbwxpm/2), doi 10.17632/v43hmbwxpm.2 | CC BY 4.0 | `Raw_Data/uottawa_bearing/` (48 files, listed in `pump_ottawa_files.json`) |
| Pump | images | XJTU-DV-Pump event camera | [Zenodo 17447083](https://zenodo.org/records/17447083) | CC BY 4.0 | `Raw_Data/xjtu_dv_pump/XJTU-DV-Pump_part1.zip`, `_part2.zip` |
| Haul Truck | sensor | University of Ottawa UODS-VAFDC, vibration and microphone | [Mendeley Data y2px5tg92h v1](https://data.mendeley.com/datasets/y2px5tg92h/1) | CC BY 4.0 | `Raw_Data/uods_vafdc/` (30 files, listed in `haul_truck_uods_files.json`) |
| Haul Truck | images | XJTU-DV-Rotor event camera | [Zenodo 17447083](https://zenodo.org/records/17447083) | CC BY 4.0 | `Raw_Data/xjtu_dv_rotor/XJTU-DV-Rotor.zip` |
| Belt Conveyor, Dozer | both | CORA rotating electromechanical system: gear-wear runs (Vx vibration, T1 temperature, infrared camera) | [CSUC Dataverse doi 10.34810/DATA2500](https://doi.org/10.34810/DATA2500) ([data descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC13254333/)) | CC BY 4.0 | `Raw_Data/cora_gear_wear/`, `Raw_Data/cora_vibration/`, `Processed_Data/cora_gear_wear/t1_bounded_samples.npz` |
| Excavator, Wheel Loader | both | CORA: drive-fault runs (bearing, unbalance, misalignment, rotor bars) | same as above | CC BY 4.0 | `Raw_Data/cora_drive_faults/`, `Raw_Data/cora_vibration/` |
| Belt Conveyor, Excavator, Wheel Loader | sensor (attempt C only) | CORA transient start-up runs: 9 classes x 4 speeds x 5 repetitions, Vx at 3 kHz and RPM at about 100 Hz for 30 s, plus one TimeVi file per class and speed (downloaded 2026-11-06) | same as above | CC BY 4.0 | `Raw_Data/cora_transient/` (396 files, 253 MB, listed with SHA-256 in `cora_transient_files.json`) |
| Drill Rig | sensor | MCC5-THU gearbox fault diagnosis datasets (Chen et al., Tsinghua University) | [Mendeley Data p92gj2732w v2](https://data.mendeley.com/datasets/p92gj2732w/2), doi 10.17632/p92gj2732w.2 | CC BY 4.0 | `Raw_Data/mcc5_thu_gearbox/` |
| Drill Rig | images | MFGNet-Gear synthetic CAD gear meshes (Mei and Shao, University of Michigan) | [Deep Blue Data doi 10.7302/qrdj-n812](https://doi.org/10.7302/qrdj-n812) | CC BY 4.0 | `Raw_Data/mfgnet_gear/mesh_ply.zip` |
| Grader | sensor | JUST Slewing bearing datasets-1, six accelerometers at 50 kHz (Xiaodie Ren, Jiangsu University of Science and Technology) | [Mendeley Data hwg8v5j8t6 v1](https://data.mendeley.com/datasets/hwg8v5j8t6/1), doi 10.17632/hwg8v5j8t6.1 | CC BY 4.0 | `Raw_Data/just_slewing/condition1.zip` … `condition6.zip` |
| Grader | images | GearBox_adddingImg v1 by Vjs, real photos of small gears with labelled teeth (Roboflow COCO export of 2026-11-05; 838 images) | [Roboflow Universe](https://universe.roboflow.com/vjs-jek7v/gearbox_adddingimg) | CC BY 4.0 | `Raw_Data/roboflow_gear_photos/GearBox_adddingImg.v1i.coco.zip` |
| Grader (matched-class experiment) | sensor | MCC5-THU healthy, tooth-break (L/M/H) and missing-tooth recordings | same as Drill Rig | CC BY 4.0 | `Raw_Data/mcc5_thu_gearbox/` (already local) |
| Grader (matched-class experiment) | sensor | Aalto Gear Failure Dataset 3012 Hz v2 (Dahl et al., Aalto University): 9:1 azimuth-thruster rig, 8 gear pairs, healthy and replicated tooth faults, 4 accelerometers, 2 torque, 5 encoders | [Mendeley Data fywnj597d8 v2](https://data.mendeley.com/datasets/fywnj597d8/2), doi 10.17632/fywnj597d8.2 ([data paper](https://doi.org/10.1016/j.dib.2024.111171)) | CC BY 4.0 | `Raw_Data/aalto_gear/AGFD_3012Hz.rar` (9,256,917,062 bytes, SHA-256 `953654381d0d834d64ce8ba1aa4775fca1a98ac1ddf8718a6b7e4fe4260504d3`, matches Mendeley; RAR5 with a 1 GB dictionary, extract with 7-Zip 23+), extracted to `Raw_Data/aalto_gear/3012Hz/` (425 runs) |

## Points to keep in mind

- Pump and Haul Truck use separate sensor and event-camera rigs. CORA sensor and infrared data share run IDs. Synchronization below run level was not verified. Drill Rig uses CAD renders and Grader compares separate tasks.
- The CORA rig has one run per class and speed. The Dozer reuses the Belt Conveyor's runs, and the Wheel Loader shares the healthy and bearing runs with the Excavator, so these mappings are not independent evidence.
- The CORA transient runs are used only by improvement attempt C (unseen repetition at a known speed). They are not part of the fair tests or the deployment checkpoints.
- The CORA T1 temperature was downloaded as a bounded sample (8 short ranges of 6,000 readings per run), not the full stream.
- UODS ball faults were recorded without load while the other classes ran at about 400 N, so the Haul Truck ball class may partly reflect load.
- The Grader has one slewing support per class, so its sensor model may partly learn the individual ring. Its images are a separate task (good vs damaged teeth of small manufactured gears, uploader unknown, many frames from 15 short videos), not the slewing ring.
- The Drill Rig images are noise-free renders of CAD meshes, not photographs, and not of the MCC5-THU gears.
- The earlier, longer dataset survey (candidate datasets that were not used) is in the Git history of this file.
