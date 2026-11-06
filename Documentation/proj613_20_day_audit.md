# PROJ 613 audit and 20-day plan

November 5, 2026 · Group 9 · Baseline `05476ba`

The saved post-push check confirms eight sensor routes, eight image routes and 19 upload/error checks. The experimental results below are separate from those checks. The routes use deployment fits, including some trained on subsets different from the archived evaluations. A working route and its prediction confidence do not establish held-out accuracy.

This review uses the requirements recorded for the capstone discussion: compare sensor and visual methods for a comparable component fault, distinguish laboratory analogues from mine equipment, preserve the split, report class-level performance and explain the methodology without code snippets. The team previously stated a target above 80% for every equipment pair. Confirm the instructor's exact metric, split and treatment of analogues before treating that target as a formal rubric rule.

## Eight equipment mappings

Scores are the archived primary held-out accuracy and macro F1, shown as percentages. The Grader image entry uses balanced accuracy. Test units are evaluation units, and several may share an underlying physical specimen.

| Equipment | Sensor accuracy / F1 | Image accuracy / F1 | Audit finding |
|---|---:|---:|---|
| Pump | 95.8 / 95.8, 24 records | 100 / 100, 24 records | Strongest aligned laboratory comparison. Both use healthy, inner race, outer race and rolling-element classes. The Ottawa and XJTU rigs are separate and use different held-out speed protocols. Verify specimen IDs before claiming generalization to new physical bearings. |
| Haul Truck | 80.0 / 74.1, 30 records | 58.3 / 56.6, 12 records | Four classes align on bearing analogues. UODS ball-fault recordings run unloaded while other classes run loaded, so load is confounded with ball fault. Sensor balanced accuracy is 73.3%. Image outer-race and ball recall are both 33.3%. Sensor holds out bearing IDs while vision holds out speed. |
| Belt Conveyor | 43.8 / 35.0, 16 runs | 50.0 / 43.2, 16 runs | Shared CORA gear-wear labels and run keys support a comparison at the experiment level. Sensor misses every wear25 run and images miss every wear75 run. The laboratory geared drive is the analogue. |
| Dozer | 43.8 / 35.0, 16 runs | 56.3 / 52.1, 16 runs | Reuses all 16 Belt Conveyor runs. Differences in visual scores do not establish a separate machine result. Report a second application mapping of the same benchmark. |
| Excavator | 25.0 / 18.3, 16 runs | 25.0 / 21.7, 16 runs | Same CORA classes across modalities, with both accuracies at the four-class baseline. Sensor bearing and misalignment recall are zero. Image healthy recall is zero. Local follow-up experiments use different targets or feature sets and need separate rows if presented. |
| Wheel Loader | 31.3 / 32.5, 16 runs | 18.8 / 20.6, 16 runs | Same CORA target across modalities. Images are below the four-class baseline and miss all broken-bar runs. Eight healthy or bearing run IDs also appear in Excavator. |
| Drill Rig | 73.3 / 70.5, 120 records | 88.9 / 88.8, 4,800 synthetic gears | The sensor study holds out speed and the image study holds out CAD designs. The image source calls R0 root breakage while the project maps it to the sensor class tooth crack. Treat that class as unaligned until justified. CAD results do not establish camera performance. Sensor healthy recall is 41.7%. [MCC5-THU data paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11070670/) · [MFGNet-Gear label definitions](https://huggingface.co/datasets/rsmei/MFGNet-Gear) |
| Motor Grader | 84.2 / 84.7, 120 records | 82.6 / 85.8, 2,879 tooth crops | Different components, labels, metrics and test units. JUST slewing-bearing tests hold out operating conditions on reused supports, one per class. The image result concerns good vs. damaged teeth and damaged recall is 66.1%. The earlier [gear-alignment feasibility review](motor_grader_gear_comparison.md) documents the missing independent physical gear IDs and full-gear label coverage. |

CORA's secondary same-recording time-split accuracy is 93.8%–100%, while the unseen-speed accuracy is 18.8%–56.3%. Each CORA row has four speeds and one run per class/speed. A single correct run changes pooled accuracy by 6.25 percentage points. Frame or window counts increase the number of observations within a run. They do not increase the number of independent test runs. The sensor selection for Belt Conveyor and Dozer also used vibration plus temperature in three of four folds, while their upload checkpoints use vibration alone. Keep those evaluation and deployment configurations distinct.

The README and methodology generalize that sensor and image data come from different rigs. CORA uses a shared rig and matching run IDs, with synchronization below the run level unverified. The methodology also describes an all-recording separation rule. That applies to the primary test, while the secondary time split deliberately reuses recordings. Correct both descriptions during implementation. [Repository results](../Results) · [Deployment verification](deployment_verification.md)

## Ranked issues and acceptance criteria

| Rank | Severity and feasibility | Work after audit review | Acceptance criteria |
|---|---|---|---|
| 1 | Critical · High feasibility · 2 team-days | Correct the comparison matrix and professor-facing claims. Mark Grader as separate tasks, Dozer as a reused benchmark, Drill Rig as synthetic and partially aligned, and all routes as current-condition classification. | Every row names the component, class mapping, modality, rig, experimental unit, split, result provenance, and applicability. Published and demo results appear in separate tables. The team confirms whether >80% is strict and whether analogues count. |
| 2 | Critical · High feasibility for reporting, moderate for improvement · 4 team-days | Freeze CORA group manifests and prioritize one new benchmark experiment before expanding. Start with a binary healthy vs. worn gear comparison on the existing shared Belt Conveyor/Dozer runs, or a reproducible four-class baseline using the same unseen-speed folds. | A predeclared target and majority baseline. Zero cross-fold run leakage. One prediction per held-out run, fold-level results, class recall, macro F1 and balanced accuracy. Report binary and four-class scores separately. Stop after the planned comparison if performance remains weak. |
| 3 | High · Moderate feasibility · 3 team-days | Quantify Haul Truck load confounding and vision limits. Verify specimen metadata, report load by class, and use a load-matched common-label subset only if both modalities support it. | Whole-bearing grouping where IDs exist, with groups and classes counted before fitting. If ball fault remains inseparable from load, flag that class and report any narrower analysis separately. |
| 4 | High · High feasibility for scope correction · 2 team-days | Check Drill Rig fault mapping against the source, and trace Pump specimen grouping. | Source-backed label crosswalk. Drill Rig root breakage and root crack remain distinct unless the source evidence supports a common target. A three-class follow-up is allowed only with the same eligible classes on both sides and the original grouped split preserved. Pump claims remain limited to the observed split. |
| 5 | Critical mismatch · Low feasibility for replacement · 1 team-day timebox | Decide whether Grader can be aligned around gear teeth. Verify full-gear photographs, physical gear IDs, compatible damage labels, reuse rights and independent groups before training. | Proceed only with complete label and group evidence for both classes on both modalities. Otherwise keep the existing two experiments and state why their scores cannot be directly compared. |
| 6 | High demo impact · High feasibility · 2 team-days | Freeze the demo manifest and practice the technical explanation. | All 16 routes load from verified local checkpoint hashes. Confirm expected ambiguity and unsupported input behavior. Show file, mapping, prediction and confidence. Demonstrate shared-format limits and explain the data-to-result flow without code snippets. |

## Schedule and decision gates

- **Days 1–3:** Review findings as a team, confirm the instructor's acceptance criteria, freeze the baseline commit, and produce the label, group and metric matrix.
- **Days 4–8:** Reconcile source IDs, archive the primary splits and make a timeboxed Grader data decision by Day 8. Prioritize the one CORA comparison with the clearest shared target.
- **Days 9–13:** If approved, run the prespecified CORA experiment and one bounded Haul Truck sensitivity analysis. Use separate output directories and keep the current deployment available. Record weak results without tuning on outer folds.
- **Days 14–16:** Review evidence and stop new modeling. Promote a candidate only if its comparable target and grouping are documented and its agreed primary score improves under the same evaluation protocol. Keep any remaining limits explicit.
- **Days 17–20:** Finalize the comparison table, methodology diagram, captions, technical narrative and demonstration. Reserve the final two days for an end-to-end run and corrections.

The audit itself made no changes to model code, weights, data or saved evaluations.
