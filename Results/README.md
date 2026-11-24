# Saved evaluation results

The [eight-equipment evidence matrix](../Documentation/equipment_evidence_matrix.md) links each equipment directory to its components, labels, grouping rules, and comparable metrics.

- `sensor.json` and `images.json` contain the archived primary evaluations. Use the sensor `score` and image `unit_level` blocks for headline results. Confusion matrices follow the recorded class order. Primary groups vary by experiment.
- `sensor_timesplit.json` and `images_timesplit.json` use earlier and later segments of the same recordings. Treat these as within-recording results and report them separately from the primary evaluations.
- Dozer and Belt Conveyor reuse the same CORA runs. The Drill Rig image results apply to CAD renders and map root breakage to the sensor root-crack class. The Motor Grader image result measures tooth condition, with balanced accuracy as the headline, and the archived primary Grader sensor result measures JUST slewing-bearing condition. The Aalto gear-sensor experiment is listed separately.
- The [deployment verification report](../Documentation/experiment_archive.md#record-phase-1-enhancement-report) records local inference smoke tests. Their confidence values belong to individual predictions from the saved deployment checkpoints.

- `experiments/` holds the November 6 follow-ups (CORA healthy vs worn, Haul Truck without ball) and `deployment/route_check.json` the route check; see the [follow-up record](../Documentation/experiment_archive.md#record-followup-experiments). `experiments/improvement/` holds the four planned improvement attempts; see the [improvement results](../Documentation/experiment_archive.md#record-improvement-results). `experiments/grader_gear/` holds the Grader healthy gear vs damaged tooth sensor results; see the [Grader matched-class results](../Documentation/experiment_archive.md#record-grader-matched-classes-results).

See the [20-day audit](../Documentation/experiment_archive.md#record-phase-1-enhancement-report) for evidence gaps and follow-up criteria.
