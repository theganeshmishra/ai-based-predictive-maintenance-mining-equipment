# Collaboration guide

Shared PROJ 613 capstone of Ganesh Mishra, Lakshya Sharma and Sagar Patel. Work in this one repository.

1. Pull the latest `main` and make a branch for your change.
2. Keep datasets, images, feature tables, models and `.venv` local (they are git-ignored). Do not commit credentials or personal paths.
3. A new machine is one new entry in `MACHINES` (section 4 of `Notebook/pdm.py`), plus a loader in section 1 if it uses a new dataset. Record its source and licence in `Dataset_Manifest/manifest.md`.
4. Keep every window or image of a test recording out of training, and make every choice (model, features, epochs) inside the training data. Report the fair test as the main result.
5. Rerun the affected `python Notebook/pdm.py ...` commands, then run the notebook, and say in the pull request which results changed.
6. Open a pull request and get a teammate's review before merging to `main`. A weak result is still useful when it is reported honestly.
