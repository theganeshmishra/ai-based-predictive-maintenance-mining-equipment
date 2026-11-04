# Methodology

## Task

For each machine, classify the **current condition** of one component into four classes, separately from a sensor recording and from camera images. The two sides use different laboratory rigs, so they are compared side by side and never combined. Nothing here predicts remaining useful life.

## Workflow (the same for every machine)

All code is in one file, [Notebook/pdm.py](../Notebook/pdm.py). Its sections follow these steps in order.

```text
raw files ──► 1a load sensor data ──► feature table (one row per time window)   ──► 3b sensor model ──┐
          └─► 1b load camera data ──► PNG images + index.csv (one row per image) ──► 3c image model  ──┤
                                                                                                      ▼
                     2  split: decides what is training and what is test           3a one answer per
                     4  MACHINES: which groups each machine holds out              recording, metrics
```

1. **Load.** Each raw recording is cut into fixed windows (100 ms or 1 s, depending on the dataset). Each window gets features computed from its own signal: statistics such as RMS, kurtosis and crest factor, and the share of energy in frequency bands. CORA bands are measured in shaft orders (frequency divided by shaft speed), so they line up across speeds. Images are exported from the raw camera files. No feature uses a label.
2. **Split.** See the two tests below. The windows or images of one recording always stay together on one side.
3. **Train.**
   - *Sensor:* five candidate models (random forest, extra trees, gradient boosting, RBF SVM, logistic regression) and the machine's feature sets. The best pair is chosen by macro F1 on validation groups held out inside the training data.
   - *Images:* an ImageNet-pretrained ResNet-18 is fine-tuned. The number of epochs (up to 15; up to 6 for gears) is chosen on a validation part of the training data, then the model is retrained on all training images.
4. **Score.** Each test recording (or gear) gets one answer: the class with the highest mean probability over its windows or images. We report accuracy and macro F1 per recording, with a confusion matrix. A majority-class control shows what always guessing the most common class would score.

## The two tests

**Fair test (main result).** Whole groups of recordings are held out, one group per round, and the model never sees any part of them:

| Machine | Sensor test | Image test |
|---|---|---|
| Pump | train speed profiles A/B, test C/D (24 recordings) | train 20 Hz, test 30 Hz (24 recordings) |
| Haul Truck | 5 rounds, each holding out 3 whole bearings (30 recordings) | train 1000 rpm, test 2000 rpm (12 recordings) |
| Belt Conveyor, Excavator, Dozer, Wheel Loader | 4 rounds, each holding out one speed (16 runs) | same |
| Drill Rig | 3 rounds, each holding out one speed (120 recordings) | 4 rounds, each holding out 3 gear designs (4,800 gears) |
| Grader | 6 rounds, each holding out one working condition (120 recordings) | each video or photo set held out in turn (tooth crops; a separate two-class task) |

Every choice (model, feature set, number of epochs) is made inside the training data only.

**Time split (secondary, easier).** Every recording is cut in time: the first 60% trains, the last 30% tests, and the 10% gap in between is dropped. The same recordings are on both sides, so this mostly measures whether the model recognises a recording it has already seen. It is reported only to show how much easier it is than the fair test. Settings are fixed (random forest on the first feature set; ResNet-18 with the same epoch rule).

## Leakage controls

- No window or image of a test recording is ever used for training or for a choice (checked in code by `check_no_overlap`).
- No normalisation uses statistics from test recordings. Images are each scaled on their own; the sensor models scale inside the training fold.
- The raw mean of the UODS accelerometer is not a feature, because inner-race files carry a DC offset.
- Seeds are fixed (sensor 42, images 42, gears and gear renders 613). PyTorch training on a multi-core CPU is not bit-for-bit repeatable, so an image rerun can differ slightly from the saved score.

## Limits

- No data comes from mining equipment. Each machine maps a mining component onto a laboratory component.
- Test sets are small. One recording is worth 6 points of accuracy on a 16-run test and 8 points on the Haul Truck's 12 image recordings.
- The CORA rig has one run per class and speed. The Dozer reuses the Belt Conveyor's runs, so it is not separate evidence.
- The Drill Rig images are noise-free CAD renders, so their score is an upper bound for a camera.
- The Grader images use a different method: frozen ImageNet ResNet-18 features and a logistic regression, with settings fixed in advance (choosing them inside the training sources is saved as a check). They show real photos of small gears, not the slewing ring, and are scored by balanced accuracy because 94% of the tooth crops are good teeth.
