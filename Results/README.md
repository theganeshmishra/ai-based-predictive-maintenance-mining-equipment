# Saved evaluation results

The [equipment evidence matrix](../Documentation/equipment_evidence_matrix.md) links each equipment directory to its components, labels, grouping rules, and comparable metrics.

- `sensor.json` and `images.json` contain the archived primary evaluations. Use the sensor `score` and image `unit_level` blocks for headline results. Confusion matrices follow the recorded class order. Primary groups vary by experiment.
- `sensor_timesplit.json` and `images_timesplit.json` use earlier and later segments of the same recordings. Treat these as within-recording results and report them separately from the primary evaluations.
- Dozer and Belt Conveyor reuse the same CORA runs. The Drill Rig image results apply to CAD renders and map root breakage to the sensor root-crack class.
- The [deployment verification report](../Documentation/deployment_verification.md) records local inference smoke tests. Their confidence values belong to individual predictions from the saved deployment checkpoints.

See the [20-day audit](../Documentation/proj613_20_day_audit.md) for evidence gaps and follow-up criteria.
