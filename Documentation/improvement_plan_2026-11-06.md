# Planned improvement attempts (written before running)

PROJ 613 · Group 9 · November 6, 2026 · committed before any of these results existed

The team asked whether more machines can reach 80% accuracy. This file fixes the attempts in advance so the results cannot be tuned to the test. Each attempt runs once. Every result is reported, whether it reaches 80% or not. The archived fair-test results and the deployment checkpoints stay unchanged; new results go to `Results/experiments/`.

## A. Drill Rig sensor: speed-normalised (order) features

- **Why:** the fair test holds out one speed (1000, 2000 or 3000 rpm). The current band features are in hertz, so gear-mesh energy moves to another band when the speed changes. Healthy recall is 5 of 12.
- **Change:** each MCC5-THU file has a `speed` channel (a tachometer pulse). For every 1 s window, the pulse rate is measured from that channel, and each vibration channel's spectrum is summed in bands of multiples of that rate (tachometer orders) instead of fixed hertz bands. The pulse rate itself is not a feature, so the model cannot use speed as a shortcut.
- **Test:** the same three folds (leave one speed out), the same five candidate models, chosen inside the training speeds. The feature sets offered are the existing `vibration` set, `orders` and `vibration+orders`.
- **Result:** one prediction per recording; accuracy, balanced accuracy and macro F1, with the majority control. It is compared with the archived 73.3%.

## B. Haul Truck images: frames that cover the same number of shaft turns

- **Why:** the image model trains on 1000 rpm recordings and tests on 2000 rpm. With fixed 125 ms frames, a test frame covers twice as many revolutions as a training frame.
- **Change:** 2000 rpm recordings are cut into 62.5 ms frames, so every frame covers about 2.1 revolutions at both speeds. The training frames (1000 rpm, 125 ms) are unchanged. The speed comes from the recording's operating condition, not from its label.
- **Test:** the same 12 training and 12 test recordings and the same fine-tuned ResNet-18 settings. The baseline (125 ms test frames) is rerun with the current code first, so both use identical code.
- **Result:** one prediction per test recording; accuracy and macro F1 for both runs. It is compared with the rerun baseline and the archived 58.3%.

## C. CORA: unseen repetition at a known speed (a different, narrower question)

- **Why:** the CORA fair test holds out a whole speed, with one stationary run per class and speed. The CORA data also has transient runs: 5 repetitions per class and speed.
- **Data:** the transient `Vx` vibration, `RPM` and `TimeVi` files from CSUC Dataverse doi 10.34810/DATA2500 (about 230 MB), saved in `Raw_Data/cora_transient/`.
- **Test:** for each CORA machine, its four classes. Five rounds, each holding out repetition k at all four speeds, so 16 test runs per round and 80 in total. The features are the existing CORA order features, normalised with the measured speed of each window when the RPM file allows it, otherwise with the nominal speed. The model is chosen inside the training repetitions. One prediction per run, with the majority control.
- **Claim if it scores well:** the model recognises a repeated run of a known class at a speed it has trained on. This is not the unseen-speed claim, and it does not replace the fair-test number. The repetitions use the same physical gear set or fault, so it is not a new-specimen test either.

## Reporting rule

All three results go into the final report with their protocol next to the number. A model is promoted to the demo only if it improves its own fair test under the same protocol (A and B). C is reported as a separate test.
