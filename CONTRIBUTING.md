# Collaboration guide

This repository is shared by Ganesh Mishra, Lakshya Sharma, and Sagar Patel for the Saskatchewan Polytechnic PROJ 613 capstone.

## Working agreement

1. Create or claim a GitHub Issue before starting material work.
2. Pull the latest `main` branch and create a focused branch.
3. Use descriptive branch names: `feature/...`, `fix/...`, `docs/...`, or `experiment/...`.
4. Keep commits small and use imperative messages, for example `Add maintenance event distribution`.
5. Open a pull request early when feedback would help; mark it as a draft until it is ready.
6. Link the issue, summarize the change, state how it was checked, and identify any data or assumptions used.
7. Obtain at least one teammate review before merging to `main`.
8. Use squash merging unless preserving separate commits has clear value.

## Notebook rules

- One person owns the single TiNA notebook at a time; record ownership in the related issue.
- Restart the kernel and run all cells before requesting review.
- Keep cells short, add Markdown interpretation, and remove accidental/debug output.
- Do not store secrets, personal paths, the TiNA data, large tables, or fitted models in notebooks.
- Move reusable functions to `src/` and add a short docstring.
- State the input data period, row count, target horizon, split dates, and random seed.
- Keep chronological order when splitting time-series data.

## Pull request checklist

- [ ] The linked issue and objective are clear.
- [ ] No raw/processed data, secrets, caches, or large model files are included.
- [ ] The notebook or code runs from a clean environment using `requirements.txt`.
- [ ] New results are reproducible and the relevant parameters are documented.
- [ ] Predictors do not use information from after the prediction time.
- [ ] Figures have titles, axis labels, units where known, and readable legends.
- [ ] The README or report is updated if the workflow or findings changed.
- [ ] At least one teammate reviewed the pull request.

## Definition of done

Work is done when it is reproducible, reviewed, merged, documented, and connected to a capstone objective. An experiment that does not improve a model can still be complete if its method and outcome are recorded clearly.

## Resolving disagreements

Discuss technical disagreements in the issue or pull request and compare alternatives against the agreed evaluation metrics, leakage risk, reproducibility, and 12-week scope. Record the final decision so it does not need to be revisited without new evidence.
