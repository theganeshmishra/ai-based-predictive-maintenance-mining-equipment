# Dataset manifest

All data are public laboratory datasets. None was recorded on mining equipment. Each machine is a mapping of one mining component onto a comparable laboratory component. Files are downloaded once by hand into `Raw_Data/` (git-ignored, never committed). `python Notebook/pdm.py data-check` lists anything missing and checks the SHA-256 where it is known.

| Machine | Side | Dataset | Source | Licence | Local folder |
|---|---|---|---|---|---|
| Pump | sensor | University of Ottawa bearing rig, vibration (Huang and Baddour) | [Mendeley Data v43hmbwxpm v2](https://data.mendeley.com/datasets/v43hmbwxpm/2), doi 10.17632/v43hmbwxpm.2 | CC BY 4.0 | `Raw_Data/uottawa_bearing/` (48 files, listed in `pump_ottawa_files.json`) |
| Pump | images | XJTU-DV-Pump event camera | [Zenodo 17447083](https://zenodo.org/records/17447083) | CC BY 4.0 | `Raw_Data/xjtu_dv_pump/XJTU-DV-Pump_part1.zip`, `_part2.zip` |
| Haul Truck | sensor | University of Ottawa UODS-VAFDC, vibration and microphone | [Mendeley Data y2px5tg92h v1](https://data.mendeley.com/datasets/y2px5tg92h/1) | CC BY 4.0 | `Raw_Data/uods_vafdc/` (30 files, listed in `haul_truck_uods_files.json`) |
| Haul Truck | images | XJTU-DV-Rotor event camera | [Zenodo 17447083](https://zenodo.org/records/17447083) | CC BY 4.0 | `Raw_Data/xjtu_dv_rotor/XJTU-DV-Rotor.zip` |
| Belt Conveyor, Dozer | both | CORA rotating electromechanical system: gear-wear runs (Vx vibration, T1 temperature, infrared camera) | [CSUC Dataverse doi 10.34810/DATA2500](https://doi.org/10.34810/DATA2500) ([data descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC13254333/)) | CC BY 4.0 | `Raw_Data/cora_gear_wear/`, `Raw_Data/cora_vibration/`, `Processed_Data/cora_gear_wear/t1_bounded_samples.npz` |
| Excavator, Wheel Loader | both | CORA: drive-fault runs (bearing, unbalance, misalignment, rotor bars) | same as above | CC BY 4.0 | `Raw_Data/cora_drive_faults/`, `Raw_Data/cora_vibration/` |
| Drill Rig | sensor | MCC5-THU gearbox fault diagnosis datasets (Chen et al., Tsinghua University) | [Mendeley Data p92gj2732w v2](https://data.mendeley.com/datasets/p92gj2732w/2), doi 10.17632/p92gj2732w.2 | CC BY 4.0 | `Raw_Data/mcc5_thu_gearbox/` |
| Drill Rig | images | MFGNet-Gear synthetic CAD gear meshes (Mei and Shao, University of Michigan) | [Deep Blue Data doi 10.7302/qrdj-n812](https://doi.org/10.7302/qrdj-n812) | CC BY 4.0 | `Raw_Data/mfgnet_gear/mesh_ply.zip` |

## Points to keep in mind

- The sensor and image sides of every machine come from different rigs or recordings, so the two scores are not a matched pair and are never combined.
- The CORA rig has one run per class and speed. The Dozer reuses the Belt Conveyor's runs, and the Wheel Loader shares the healthy and bearing runs with the Excavator, so these mappings are not independent evidence.
- The CORA T1 temperature was downloaded as a bounded sample (8 short ranges of 6,000 readings per run), not the full stream.
- UODS ball faults were recorded without load while the other classes ran at about 400 N, so the Haul Truck ball class may partly reflect load.
- The Drill Rig images are noise-free renders of CAD meshes, not photographs, and not of the MCC5-THU gears.
- The earlier, longer dataset survey (candidate datasets that were not used) is in the Git history of this file.
