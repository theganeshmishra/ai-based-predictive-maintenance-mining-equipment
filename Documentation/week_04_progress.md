# Week 4 capstone progress checkpoint

**As of October 5, 2026 — Saskatchewan Polytechnic PROJ 613.** This is a
research-prototype checkpoint, not a claim of deployment-ready predictive
maintenance or eight completed equipment models.

## Completed by Week 4

- Identified eight equipment categories and locked the
  [component/fault dataset matrix](../Dataset_Manifest/manifest.md), including
  source access, labels, licences, sample units, and transfer limitations.
- Completed the [Pump support-bearing sensor-versus-event-image experiment](../Results/pump_sensor_visual_comparison.md)
  on healthy, inner-race, outer-race, and ball-fault conditions. The sensor
  model scored **0.958 accuracy / 0.958 macro F1** on 24 held-out recordings;
  the visual model scored **1.000 / 1.000** on 24 held-out recordings. A harder
  held-out camera-configuration visual check scored **0.854 accuracy / 0.855
  macro F1**. Different rigs and operating shifts prevent a claim that one
  modality is better; neither branch uses mining-pump repair histories.
- Completed the [Belt Conveyor geared-drive analogue sensor-versus-infrared experiment](../Results/belt_conveyor_sensor_visual_comparison.md)
  on the same 16 CORA laboratory runs, with each speed held out in turn.
  Healthy versus gear-worn T1 temperature scored **0.625 accuracy / 0.564 macro
  F1**; infrared/OpenCV scored **0.500 / 0.418**. Both accuracies are below the
  **0.750 majority-class control**. This is a completed comparison with weak
  generalization, **not** a successful fault detector or a mining-conveyor
  validation. The [run-level metrics](../Results/belt_conveyor_cora.json) and
  [confusion matrices](../Figures/belt_conveyor_cora_confusions.png) preserve
  the measured findings.

## Planned, deliberately not implemented yet

| Equipment category | Mapped component/fault | Week 4 status |
|---|---|---|
| Haul / dump truck | Electric-drive wheel-motor bearing | Dataset pair mapped; no experiment |
| Excavator | Swing-drive gear wear | CORA component analogue mapped; no excavator experiment |
| Dozer | Final-drive gear wear | Same CORA analogue mapped; no dozer experiment |
| Wheel loader | Final-drive/transmission gear wear | Same CORA analogue mapped; no loader experiment |
| Drill rig | Rotary-head reduction gear wear | Same CORA analogue mapped; no matched visual experiment; existing PHM drill-pressure work is sensor-only and a different fault |
| Grader | Circle-drive gear wear | Same CORA analogue mapped; no grader experiment |

The five remaining CORA mappings reuse the **same laboratory source** as the
Belt Conveyor analysis. They are not five new datasets or automatically
completed equipment validations. Equipment #3 has not started.

## Limits and later-week plan

The current models classify **present component condition**. They do not
predict a future repair date or remaining useful life. There is no mining-fleet
holdout, no independently tested second CORA gearbox, and no justified
sample-level sensor/image fusion. The Pump modalities use different physical
rigs; the Belt Conveyor result is from a single lab geared-drive system.

For later weeks, first review the Pump and Belt Conveyor evidence and transfer
claims with the professor. If the component-level scope is accepted, choose
and pre-register one of the six remaining categories, verify a genuinely
comparable fault target and independent test units, then model that category
without reusing the Belt Conveyor score as a new equipment result. No further
equipment modelling is part of this Week 4 checkpoint.
