# Collaboration guide

This is the shared Saskatchewan Polytechnic PROJ 613 capstone of Ganesh Mishra, Lakshya Sharma, and Sagar Patel. Work in this **one repository** and follow the [active project README](README.md).

1. Pull current `main`, claim or create a GitHub Issue, and make a focused branch.
2. Record dataset provenance, licence, comparable component/fault labels, and independent test units in the [matrix](Dataset_Manifest/manifest.md) before new modelling.
3. Keep downloaded datasets, raw images/events, processed tables, virtual environments, caches, and model binaries local. Do not add credentials or personal filesystem paths.
4. Use group-aware held-out evaluation. Keep frames/windows from a test run entirely out of training, and report per-class metrics and confusion matrices.
5. Distinguish a shared laboratory component benchmark from data collected on the named mining equipment. Do not pair samples from unrelated physical machines or claim future-repair prediction without future outcome labels.
6. Update the methodology, results, and progress checkpoint when an experiment changes. The six remaining equipment categories are planned, not completed.
7. Open a pull request with the issue, files changed, data prerequisites, checks run, and limitations. Obtain teammate review before merging to `main`.

The [root dependency list](requirements.txt) and VS Code tasks are the supported local entry points. A negative or weak result is still useful when the evaluation and limitation are preserved accurately.
