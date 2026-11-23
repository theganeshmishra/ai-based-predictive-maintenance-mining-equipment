# Recording script: Lakshya Sharma

Role: Developer and tester. About 9 minutes of reading in total.

Read [README.md](README.md) first. Save **one audio file per chapter**, named exactly as shown. Leave about one second of silence between numbered lines, so each slide can be matched to your voice. If you stumble, pause and re-read the whole line; do not stop the recording.

## Chapter 2 of 30: Project team and roles (your part)

Save as: `02_Project_team_and_roles_Lakshya.m4a` (or .wav / .mp3)

1. I am Lakshya Sharma, a developer and the team's tester.

## Chapter 9 of 30: Check the local checkpoints

Save as: `09_Check_the_local_checkpoints.m4a` (or .wav / .mp3). About 0.8 min.

1. Before showing predictions, the notebook displays the checkpoint status.
2. The application has eight main sensor routes and eight main image routes, plus the additional Aalto gear sensor route for Motor Grader.
3. A checkpoint can be present on disk without being trusted, and a trusted filename alone does not prove that a real prediction will work.
4. The status table is therefore only the first check.
5. It is followed by actual inference calls.
6. The local verification found all required checkpoint files and their registrations.
7. The saved models remain unchanged during this video preparation.
8. The code archive deliberately excludes their large binaries, so someone running a fresh copy needs the documented local assets as well as the Python dependencies.

## Chapter 18 of 30: Notebook upload interface

Save as: `18_Notebook_upload_interface.m4a` (or .wav / .mp3). About 0.9 min.

1. The notebook upload interface uses a file-upload widget, a mapping dropdown and a prediction button.
2. The upload accepts supported sensor recordings or image formats, but format acceptance does not guarantee source compatibility.
3. The callback first checks whether a file was actually supplied.
4. It then passes the temporary input to the same application prediction function.
5. Automatic routing is useful only where the source can be established.
6. The dropdown remains necessary for shared laboratory formats and the explicit gear-tooth inspection route.
7. The interface should display either a condition prediction with its model information or a clear explanation of why the input cannot be scored.
8. A user confirmation selects an application mapping; it does not turn laboratory data into real measurements from that mining machine.

## Chapter 19 of 30: Executed Pump sensor example

Save as: `19_Executed_Pump_sensor_example.m4a` (or .wav / .mp3). About 0.8 min.

1. This panel shows the actual output of a Pump sensor inference call using an available local recording.
2. The application identifies a compatible bearing source, extracts the expected features and loads the trusted sensor checkpoint.
3. The predicted class and probability are returned with the model name.
4. The example is useful for demonstrating that the complete path works from file to answer.
5. It must not be described as a fresh held-out accuracy measurement.
6. The route-check samples can be fitting data, and the archived evaluation uses its own group separation.
7. The application mapping is Pump, while the source remains an Ottawa bearing-rig measurement.
8. That distinction keeps the demonstration truthful without reducing the value of showing a real model execute successfully.

## Chapter 20 of 30: Executed Pump image example

Save as: `20_Executed_Pump_image_example.m4a` (or .wav / .mp3). About 0.9 min.

1. The Pump image example uses a local event-camera frame with known provenance.
2. The model returns its current-condition classification and class probabilities.
3. An event frame represents changes observed by a dynamic-vision sensor; it is different from a standard RGB photograph of a pump.
4. Its supported size and channel pattern help identify the source family, while local source indexing can establish a unique mapping.
5. The sensor and image examples are separate observations, not synchronized recordings of one machine.
6. Showing them beside each other demonstrates two independent prediction routes.
7. It does not demonstrate fusion, causal superiority of one modality, or the ability to detect an internal bearing fault from an arbitrary photograph.
8. Those stronger claims would require additional matched evidence.

## Chapter 21 of 30: Executed Motor Grader gear sensor

Save as: `21_Executed_Motor_Grader_gear_sensor.m4a` (or .wav / .mp3). About 0.9 min.

1. Motor Grader needs special explanation because two component tasks are retained.
2. The original sensor route diagnoses slewing-bearing conditions.
3. The aligned comparison adds an Aalto gear sensor checkpoint for a broad healthy-versus-damaged gear target.
4. Here the actual local Aalto recording produces a healthy prediction with a probability of zero point nine eight seven.
5. This is a deployment response on a fitting recording.
6. It is not ninety-eight point seven percent held-out accuracy.
7. The independent gear-pair experiment is the evidence used to discuss generalization.
8. The Aalto dataset is a laboratory gearbox analogue, not a field recording from a Motor Grader.
9. Preserving both routes avoids deleting working evidence while making the component comparison more mechanically defensible.

## Chapter 22 of 30: Executed Motor Grader tooth image

Save as: `22_Executed_Motor_Grader_tooth_image.m4a` (or .wav / .mp3). About 0.8 min.

1. The Motor Grader image route scores a real tooth inspection crop using the saved feature-based classifier.
2. The example panel shows the actual damaged-tooth prediction and the fields returned by that implementation.
3. The model confidence display is still an uncalibrated output and should not be interpreted as certainty of real equipment failure.
4. Sensor and image tasks share a broad gear-condition target, but one label is attached to a gear-pair run and the other to a local tooth crop.
5. Their physical gears are different.
6. The existing tests therefore support a comparison of related classification tasks, not a paired same-grader experiment.
7. The original bearing checkpoint remains available independently and should not be substituted into this gear-tooth comparison.

## Chapter 23 of 30: Ambiguous equipment is a valid result

Save as: `23_Ambiguous_equipment_is_a_valid_result.m4a` (or .wav / .mp3). About 0.9 min.

1. This execution supplies a compatible CORA recording without selecting an equipment mapping.
2. The expected behaviour is an ambiguous equipment response listing the supported candidates.
3. The system cannot infer that an identical laboratory source must belong to one particular whole mining machine.
4. This is a reliability feature, not a missing classification trick that should be bypassed with a guess.
5. Manual mapping confirmation allows a known task to be evaluated, but it does not create new evidence about the original physical machine.
6. The same principle applies to image provenance.
7. A visually similar frame can remain ambiguous when its source index cannot establish a unique route.
8. The final demonstration should explain this behaviour clearly rather than presenting every upload as automatic identification.

## Chapter 24 of 30: Invalid inputs and errors

Save as: `24_Invalid_inputs_and_errors.m4a` (or .wav / .mp3). About 0.8 min.

1. The invalid-input examples test ordinary text, an unrecognized CSV and corrupted image bytes.
2. Each example was actually sent to the prediction function, and the displayed response records the resulting rejection.
3. These checks matter because an upload control can accept a filename even when the content is unusable.
4. The model-loading path also verifies checkpoint trust and expected metadata.
5. A missing or untrusted checkpoint must not be replaced with a random estimator or a fabricated prediction.
6. Errors should explain why the input could not be scored, while leaving the working application stable.
7. These examples demonstrate handling of specific tested cases; they are not a guarantee that every imaginable malformed file has been exhausted.

## Chapter 25 of 30: Full route-check evidence

Save as: `25_Full_route_check_evidence.m4a` (or .wav / .mp3). About 0.9 min.

1. The complete route check executed thirty-nine saved-model examples successfully, covering all eight sensor and image mappings plus the additional Aalto gear route.
2. Unsupported-file rejection and shared-CORA ambiguity checks also passed.
3. The important qualification is the definition of pass: a model loaded, accepted a compatible input and returned the required output.
4. Two Drill Rig images were classified as healthy when their source labels were gear wear and gear pitting.
5. Those cases remain visible as prediction errors even though execution succeeded.
6. The clean publication checkout passed two hundred fifty-five automated tests, with two tests skipped because local data were not available.
7. These counts come from earlier verification runs; the test suite was not rerun while this video was prepared.

## Chapter 27 of 30: Time split versus held-out groups

Save as: `27_Time_split_versus_held_out_groups.m4a` (or .wav / .mp3). About 0.9 min.

1. Supplementary time-split checks use the first sixty percent of a recording for training, discard a ten percent gap and test on the last thirty percent.
2. Several saved results are much higher than the unseen-speed results.
3. This is not a reason to hide the more difficult evaluations.
4. The two protocols ask different questions.
5. A later segment of a known recording can retain the same speed, specimen and acquisition signature as training.
6. A gap reduces direct temporal overlap, but it does not create a new machine or operating condition.
7. The report labels these scores separately and notes that fitting settings and sample pools can differ.
8. No score from this procedure establishes future failure forecasting, and no unsupported time-split result is assigned to a mapping without a saved experiment.
