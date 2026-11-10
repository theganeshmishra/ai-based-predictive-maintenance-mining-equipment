# Demonstration script

PROJ 613 · Group 9 · November 10, 2026 · about 10 minutes

The files below are the local samples used by `route-check` on November 10, 2026 (`Results/deployment/route_check.json`). They are training samples, so the demo shows that each route works; the accuracy claims come from the fair tests in the [final report](final_report.md).

## Before the demo (on the demo computer)

1. Pull the latest `main` and install packages: `.venv\Scripts\python.exe -m pip install -r requirements.txt`.
2. Make sure `Processed_Data/models/` holds the checkpoints and that you built them yourself, then run `.venv\Scripts\python.exe Notebook\pdm.py trust-models` once.
3. Run `.venv\Scripts\python.exe Notebook\pdm.py route-check`. Every route with a local sample should say PASS. If one says "not in the trusted model list", repeat step 2.
4. Open `Notebook/Integrated_Equipment_Fault_Prediction.ipynb`, select the `.venv` kernel and run all cells. The first table shows which checkpoints are found.

## Script

| Step | Do | Expected answer | Say |
|---|---|---|---|
| 1 | Show the [method diagram](../Figures/methodology_flow.png) | – | Features come only from the signal or image; whole groups are held out; one answer per recording. |
| 2 | Upload `Raw_Data/uottawa_bearing/I-C-1.mat`, Automatic | Pump, inner race, 99.9% | The format check finds the Ottawa rig, so the Pump model is used. Pump is our strongest fair-test result: 95.8% sensor, 100% images. |
| 3 | Upload `Images/pump/outer/pump_outer_20Hz_15_-30_t000.png`, Automatic, then choose Pump | first **Ambiguous Equipment** (Pump or Haul Truck), then Pump, outer, 100% | Pump and Haul Truck use the same event-camera format, so the system asks which machine; it does not guess. |
| 4 | Upload `Raw_Data/uods_vafdc/B_11_2.mat` | Haul Truck, ball, 99.0% | The Haul Truck sensor fair test is 80.0%, but ball faults were recorded without load, so we report it with that caveat. |
| 5 | Upload `Raw_Data/cora_vibration/W50_F50_S_Vx.csv`, Automatic | **Ambiguous Equipment** (Belt Conveyor, Excavator, Dozer, Wheel Loader) | This is intended: one CORA rig stands in for four machines, so the system will not guess which one. |
| 6 | Same file, choose Belt Conveyor | Belt Conveyor, wear50, 100% | With a confirmed mapping it predicts. On an unseen speed CORA scores only 44%, which is our main weakness. |
| 7 | Upload `Processed_Data/predict_examples/gear_pitting_2000rpm_20Nm.csv` | Drill Rig, gear pitting, 97.7% | Drill Rig images are CAD renders, 88.9% on unseen designs; sensor 73.3%, 77.5% with order features. |
| 8 | Upload `Processed_Data/predict_examples/just_outer_ring_condition3.csv` | Grader, outer ring, 94.1% | Grader slewing-bearing sensor model, 84.2% on unseen working conditions. |
| 9 | Upload `Raw_Data/aalto_gear/3012Hz/250RPM/0_12Nm/Healthy/GP1.csv` | Grader gear task, healthy, 98.7% | The matched-class gear model: 77.8% balanced on gear pairs it never saw. It is a gear-pair analogue, not the slewing bearing. |
| 10 | Upload `README.md` | rejected: matches no known format | Unknown inputs are refused, not guessed. |
| 11 | Show the [80% scorecard](final_report.md#6-which-machines-reach-80) | – | Four of eight machines reach 80% under the fair test; the four CORA machines do not, and we explain why instead of changing the test. |

## Known surprises (do not hide them)

- On the route-check samples, the Excavator misalignment image is predicted **bearing** (85.2%) and the Drill Rig pitting render is predicted **healthy** (68.7%). Both are training samples. Say so if they come up.
- The Wheel Loader sensor sample `HB_F50_S_Vx.csv` is correct but with only 42.1% confidence.
- The Grader tooth-photo route needs `grader_gear_tooth.joblib` on the demo computer; without it the notebook reports "Model unavailable".

## Likely questions

- **Is this mining data?** No. Every source is a public laboratory or synthetic dataset mapped onto a mining component.
- **Why not fuse sensor and image?** The two sides come from different rigs (except CORA, where only run IDs match), so there are no paired samples to fuse.
- **Why not 80% everywhere?** The CORA rig has one run per class and speed; holding out a speed leaves the model with a speed it has never seen. At a trained speed it scores 97.5 to 100%, but that is an easier test and we label it as one.
- **Can it predict when a part will fail?** No. It classifies current condition only.
