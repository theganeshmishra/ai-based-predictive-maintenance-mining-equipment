# Week 7 (proposal, started early): anomaly detection

**October 2026 — Saskatchewan Polytechnic PROJ 613.** The proposal's Week 7 deliverable is
"Isolation Forest / One-Class SVM anomaly results". A first version was built while completing Week 6.
This is an early exploratory work record, **not part of the five-machine Week 6 checkpoint** or evidence that the remaining three equipment mappings have been evaluated.

## What was done

- New command for every machine: `python Models/<machine>.py anomaly` (add `--split time` for the secondary
  test). The single equipment notebook has a matching section for each machine.
- **Isolation Forest** and **One-Class SVM** learn only what *healthy* sensor data looks like. No fault labels
  are used for training. Same CSV features and the same held-out recordings as the Week 6 main test.
- Each test recording gets one score (mean of its window scores). The alarm threshold is fixed **before
  testing**: each healthy training recording is scored by a model fitted on the other healthy training
  recordings, and the highest of those scores is the threshold.
- Reported: **ROC-AUC** (how well the score separates healthy from faulty without any threshold;
  0.5 = guessing), **fault detection rate** and **false alarm rate** at the fixed threshold.

## Results (main test)

| Machine | Detector | ROC-AUC | Faults detected | False alarms |
|---|---|---:|---:|---:|
| Pump (24 recordings) | Isolation Forest | **0.98** | **100%** | 17% |
| | One-Class SVM | 0.94 | 100% | 17% |
| Haul Truck (30) | Isolation Forest | **0.96** | 67% | **0%** |
| | One-Class SVM | 0.93 | 40% | 7% |
| Belt Conveyor (16) | Isolation Forest | 0.63 | 0% | 0% |
| | One-Class SVM | 0.58 | 33% | 25% |
| Excavator (16) | Isolation Forest | 0.42 | 0% | 0% |
| | One-Class SVM | 0.50 | 17% | 25% |

Time split (secondary): ROC-AUC Haul Truck 0.996 / 1.00, Belt Conveyor 0.92 / 0.77, Excavator 0.88 / 0.73
(Isolation Forest / One-Class SVM).

## What it means

- **Pump and Haul Truck:** without ever seeing a fault, the detectors rank faulty recordings above healthy
  ones very well (ROC-AUC 0.93–0.98). The Pump detectors catch all 18 faulty recordings with 1 false alarm
  out of 6 healthy ones. The conservative Haul Truck threshold gives no false alarms (0 of 15) but misses
  faults: Isolation Forest catches 80% of inner, 100% of outer and only 20% of ball faults (ball faults were
  recorded without load, which may make them look closer to normal).
- **Belt Conveyor and Excavator:** at an unseen speed the detectors are near guessing. The healthy
  training data come from three other speeds, so a new speed already looks "unusual". Within the same
  recordings (time split) the separation is much better (ROC-AUC 0.73–0.92).
- Small test sets (16–30 recordings): one recording changes a rate by 3–8 points.

## Next

Proposal Week 8 is the OpenCV/CV baseline; anomaly thresholds and a per-speed healthy reference for the CORA
machines can be revisited if time allows.
