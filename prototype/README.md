# Prototype demo

A small local web app (Streamlit) that shows the capstone pipeline on recorded data. **No physical sensors are needed**: stored sensor readings and photos are replayed as a "virtual sensor".

It has five tabs:

| Tab | Data | Classes | What it shows |
|---|---|---|---|
| **📷 Check a photo** | your own photos, the camera, or built-in samples | 3 wear classes / 4 bearing cues | Upload one or more photos and see the whole inspection pipeline (details below) |
| Overview | all datasets | | The master catalogue (one row per sample from every dataset) and an honest results table |
| Sensor states | TiNA mining machine, September 2017 | **8** | Minute-by-minute replay of unseen days: true state vs predicted state, sensor traces, probabilities, evaluation |
| Wear inspection | Nonastreda milling tools | 3 | Per sample: force signals, tool/chip/surface photos, predicted wear from a model that never saw that tool |
| Bearing photos | owner-supplied bearing photos | 5 heuristic | Photo groups from an OpenCV heuristic; no model claimed |

The datasets come from different equipment and share no ids, columns or labels, so they are **listed together but never merged into one training table** (proposal slide 10).

## Check a photo

You can upload JPG or PNG files (several at once), take a picture with the webcam, or pick built-in samples. Each photo goes through these steps:

1. **Photo type.** Pick it, or let the app auto-detect it. Auto-detect chooses the reference set (tool edge, chips, machined surface, bearing) that the photo is closest to.
2. **OpenCV preprocessing, shown step by step.**
   - Wear photos: contrast enhancement (CLAHE), Canny edges and a texture heat map.
   - Bearing photos: an overlay marking the pixels behind each damage cue (rust hue, strong colour, very dark deposits), with the blue table dimmed.
3. **Result.**
   - Wear photos: a random forest trained on that photo type gives sharp / used / dulled scores. For tool-edge photos the app also plots the row brightness profile.
   - Bearing photos: no verified labels exist, so the app shows the OpenCV screening rule instead. It gives each cue's percentile against the 1,841 reference photos and flags any cue above the 70th percentile.
4. **Five most similar reference photos**, with their true labels (and measured flank wear for wear photos).
5. **Familiarity check.** If a photo is further from every reference photo than 99% of reference photos are from their nearest neighbour in another tool or capture session, the app marks the result as unreliable. In testing, random noise and a plain grey image were flagged, and real photos were not.
6. **Summary table**, downloadable as a CSV report.

Built-in samples, and uploaded copies of dataset photos, are judged by the model trained *without that tool*. Their result is therefore a real test result, and the true label is shown next to it.

How good is each photo type? This is tested on tools the model never saw (`results/prototype_image_models_cv.csv`):

| Photo type | Macro-F1 (mean ± std, 10 tools) |
|---|---:|
| Machined surface | 0.83 ± 0.07 |
| Metal chips | 0.81 ± 0.05 |
| Tool edge | 0.45 ± 0.14 (weak; the app says so) |

Chance is about 0.33.

Limits:
- These are microscope images from one milling machine. A phone photo of other equipment will usually be flagged as unfamiliar, and should be.
- Blurring a chip photo was enough to change its prediction, so image quality matters.

## The 8 classes

The professor asked for an 8-class demo. No 8-class label exists in the data, so the team defined 8 **machine states per minute** from TiNA's own `m_id`, `m_subid` and `alarms` columns:

| # | State | Rule | Minutes | Maintenance episodes |
|---|---|---|---:|---:|
| 1 | `normal` | no maintenance, no alarm in the minute | 24,262 | |
| 2 | `alarm` | an alarm, but no maintenance | 3,816 | |
| 3 | `M4/MS3` | maintenance code M4, subtype MS3 | 2,317 | 16 |
| 4 | `M4/MS5` | M4 / MS5 | 526 | 9 |
| 5 | `M9/MS1` | M9 / MS1 | 245 | 10 |
| 6 | `M9/MS6` | M9 / MS6 | 114 | 9 |
| 7 | `M2` | M2 (any subtype) | 134 | 3 |
| 8 | `other_maintenance` | M4/none, M7, M11, M13 (1 to 2 episodes each) | 92 | 5 |

That's 31,506 minutes from 25 days (6 to 30 Sep 2017). A minute is kept only if it has at least 30 seconds of data. Features are the mean and standard deviation of the 103 numeric sensors in that minute; `m_id`, `m_subid` and `alarms` are never used as inputs. This definition is a team choice, not part of the dataset, so change it in `build_master.py` (`STATES`) if the professor meant something else.

## Results (all from `results/prototype_8state_*.csv`)

| Evaluation | Balanced accuracy | Macro-F1 | Maintenance episodes recognised |
|---|---:|---:|---:|
| 5 blocks of whole days held out | 0.126 | 0.111 | 0 of 52 |
| Train 6 to 22 Sep, test 23 to 30 Sep | 0.167 | 0.149 | 0 of 10 |
| **Leaky** random-minute split (comparison only) | 0.946 | 0.935 | n/a |

Chance for 8 classes is 0.125. **On days it has not seen, the model predicts `normal` for almost every minute and recognises none of the maintenance episodes.** The same model looks about 95% accurate when minutes are split at random, because neighbouring minutes of the same day land in both training and test data; it learns *which day and time* a minute comes from, not the machine state. A drift-corrected version (each minute compared with the previous two hours) was also tried and stayed at chance.

This is a real finding about the data and the features, not a bug. Note also that the last 8 days contain only 6 of the 8 states (`M9/MS6` and `M2` do not occur).

The wear-inspection tab uses the Nonastreda model from the sibling project `multimodal-pdm-from-scratch`. Under leave-one-tool-out it reaches macro-F1 of about 0.84 using signals and photos together (0.86 pooled accuracy). That is milling-tool wear, not mining equipment.

## How to run (Windows PowerShell, from the repository root)

```powershell
.venv\Scripts\python.exe -m pip install -r prototype\requirements.txt
.venv\Scripts\python.exe prototype\build_master.py        # master catalogue + TiNA 8-state table (about 1 min)
.venv\Scripts\python.exe prototype\train_8state.py        # evaluation + demo predictions (a few minutes)
.venv\Scripts\python.exe prototype\export_nonastreda.py   # wear-inspection predictions (needs the sibling project)
.venv\Scripts\python.exe prototype\train_image_models.py  # photo-check models (about 1 min)
.venv\Scripts\python.exe -m streamlit run prototype\app.py
```

Then open http://localhost:8501. Once the data files are built, you can also just double-click `prototype\run_demo.bat`. Keep its window open while you use the demo.

You need:
- `data/interim/tina_2017-09_combined.parquet` (TiNA, 6 to 30 Sep 2017)
- `data/processed/bearing_image_pseudo_labels.csv` and `opencv_photos_all_in_one.parquet` (from the OpenCV notebooks)
- for the wear tab, `../multimodal-pdm-from-scratch` with its notebooks 01 and 02 run (or set `NONASTREDA_PROJECT` / `NONASTREDA_DIR`)

All data and generated files stay local under `data/` (git-ignored). Seed 613 everywhere.

## Limitations

- Academic prototype, not production-ready; no live sensor feed and no safety decisions.
- TiNA sensors are anonymised, so no physical meaning can be given to any feature.
- The 8-state classes are a team definition. Several states have very few episodes (M2: 3, other_maintenance: 5), so even a good model could not be judged reliably on them.
- The wear-inspection data is from a milling machine, and the bearing photos have unverified labels and unconfirmed permission to share. Neither is evidence about mining equipment.
- Nonastreda: Truchan, H., and Ahmadi, Z. (2025). *Nonastreda: Multimodal Dataset for Identifying Tool Wear Condition.* Mendeley Data, V1. https://doi.org/10.17632/m892d2wtzh.1 (CC BY 4.0).
