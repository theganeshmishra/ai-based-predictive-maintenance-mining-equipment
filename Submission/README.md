# Final capstone submission

AI-Based Predictive Maintenance for Mining Equipment · Group 9

Ganesh Mishra, Lakshya Sharma and Sagar Patel. Supervisor: Dr. Himanshu Patel.

This folder shares the completed submission materials and the existing software with the team. It is a snapshot based on source revision `5ffc2638f91d14229928d1c3e825cef03b76d0b3`. No trained models or experimental scores were replaced while preparing these files.

## Files

| Material | File |
|---|---|
| Final report | [PDF](Group9_Final_Capstone_Report.pdf) and [editable Word version](Editable/Group9_Final_Capstone_Report.docx) |
| Explained, executed notebook | [Group9_Capstone_Submission.ipynb](Group9_Capstone_Submission.ipynb) |
| Research poster | [96 inch wide by 48 inch high PDF](Group9_Capstone_Poster_96x48.pdf) |
| Animated walkthrough | [27 minute 7 second MP4](Video/Group9_Capstone_Walkthrough.mp4) |
| Interactive walkthrough | [Standalone HTML](Video/Group9_Interactive_Walkthrough.html) |
| Narration captions | [English SRT](Video/Group9_Video_Captions.srt), also selectable in the MP4 |
| Narration text | [Transcript](Video/Group9_Video_Transcript.txt) |
| Navigation | [Chapter timestamps](Video/Group9_Video_Chapters.txt) |

The poster follows the instructor's section list: title and objective, block diagram, dataset, models, results, use-cases, contributors and references. It shows held-out balanced accuracy only; sensor and image scores come from different rigs and protocols and are not fused. The report retains all numerical results and reports the within-recording time split only as a supplementary check.

## Access for the team

Use this existing repository, rather than creating another repository or a fork:

```bash
git clone https://github.com/theganeshmishra/ai-based-predictive-maintenance-mining-equipment.git
cd ai-based-predictive-maintenance-mining-equipment
```

For an existing clean clone:

```bash
git switch main
git pull --ff-only origin main
```

Save or commit local edits before switching or pulling. If Git reports a conflict or divergent history, stop and review it; do not use a hard reset to discard work. Older clones from before the history cleanup may need a fresh clone, keeping the old directory intact.

GitHub provides a Download raw file option for binary files. For the HTML companion, download the file and open it in a desktop browser. It is self-contained and does not need a public hosting service. Click Play narration; installed voices vary by browser and device. GitHub's source viewer does not run JavaScript or browser speech.

## Running real predictions

The submission notebook imports [the existing application module](../Notebook/predictive_maintenance.py) from the repository root. It does not need the separate code ZIP used in the local submission package. Keep this folder inside the clone so its parent-folder search can find the module. The repository's [requirements](../requirements.txt) and [locked package list](../requirements-lock.txt) are shared rather than duplicated here.

On Windows, from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install ipykernel
```

Open `Submission/Group9_Capstone_Submission.ipynb` in VS Code and select `.venv\Scripts\python.exe` as its Python kernel. On macOS/Linux use `python3` and `.venv/bin/python` instead.

For live predictions, arrange the verified source assets separately and preserve their existing structure:

- `Raw_Data/`: compatible original sensor and event/infrared sources.
- `Images/`: exported component images and their provenance indexes.
- `Processed_Data/models/`: eight sensor checkpoints, eight image checkpoints and the additional Grader gear checkpoint, with `trusted_sha256.json`.
- Other documented preprocessing caches only when a specific route requires them.

See the [dataset manifest](../Dataset_Manifest/manifest.md) for sources and licensing, and the [deployment instructions](../README.md#quick-start) for the existing workflow. Copy only checkpoints obtained through a verified handoff; retain the SHA-256 registry and do not register arbitrary downloaded models just to bypass a warning.

After verified assets are in place:

```powershell
.venv\Scripts\python.exe Notebook\predictive_maintenance.py route-check
.venv\Scripts\python.exe Notebook\predictive_maintenance.py serve
```

Open `http://localhost:8613` for the local upload page. Stop it with Ctrl+C. Without assets, the notebook can explain the workflow and display archived results, but it reports unavailable inference inputs/checkpoints. It must not be presented as a successful real-file prediction test on that machine.

## Verification before publishing

The clean publication checkout passed 255 automated tests with two local-fixture skips and 18 existing deprecation warnings; lint passed. All 11 notebook code cells executed from a clean kernel without local inference assets. That portability run correctly reported unavailable inputs and zero executed inference routes, rather than claiming a real prediction pass. The delivered notebook retains the previously verified real-asset outputs.

## Verification and interpretation

The November 22 local run completed 39 real-file inference executions and the invalid-file and CORA ambiguity checks. Two Drill Rig images returned healthy instead of their source wear/pitting labels. Execution PASS means that the route returned an output, not that its classification was correct. Motor Grader Aalto's healthy probability of 0.987 is a fitting-recording deployment output, not held-out accuracy.

Earlier full project verification recorded 257 passing local tests and 255 passing hosted tests with two expected local-data skips. These are historical records for the pinned source revision. The commit adding this folder must be assessed against its own CI result.

The animated video is computer narrated, with recorded model outputs replayed for explanation. It is not a live screen recording or team-member voice recording. The HTML companion likewise does not run Python or accept live prediction uploads. Browser end-event progression was checked with controlled speech callbacks; the controls passed 11 interaction checks with no browser exceptions. MP4 audio/video fully decoded without errors. Review the deliverables against the instructor's submission instructions.

The datasets contain laboratory component analogues and synthetic CAD renders, rather than field validation across eight mining machines. CORA mappings share benchmarks; Dozer reuses the conveyor task. Sensor and image samples remain independent, with no paired multimodal fusion. Current fault classification does not establish a failure date or remaining useful life.

## Files deliberately excluded

Raw and processed datasets, exported dataset images, original event recordings, fitted model binaries, virtual environments, caches and temporary files remain ignored. The reviewed walkthrough MP4 is a presentation artifact, not a raw dataset, and is the only video exception added to `.gitignore`. Duplicate submission/code ZIP archives are not committed because their contents are already available as repository files.
