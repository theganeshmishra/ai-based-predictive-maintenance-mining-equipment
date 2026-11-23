# Recording script: Sagar Patel

Role: Planner and developer. About 10 minutes of reading in total.

Read [README.md](README.md) first. Save **one audio file per chapter**, named exactly as shown. Leave about one second of silence between numbered lines, so each slide can be matched to your voice. If you stumble, pause and re-read the whole line; do not stop the recording.

## Chapter 2 of 30: Project team and roles (your part)

Save as: `02_Project_team_and_roles_Sagar.m4a` (or .wav / .mp3)

1. I am Sagar Patel, the planner and a developer.
2. Our project was supervised by Doctor Himanshu Patel, and the same roles are listed in the final report and on the poster.

## Chapter 8 of 30: Open the integrated notebook

Save as: `08_Open_the_integrated_notebook.m4a` (or .wav / .mp3). About 0.9 min.

1. The submitted notebook is the main explanation and demonstration entry point.
2. Its first code cell imports the standard libraries, numerical packages, plotting tools and upload widgets.
3. It locates the project folder and imports predictive maintenance as the application module.
4. The implementation is kept in that companion Python file so the notebook does not duplicate thousands of lines of model and interface logic.
5. The notebook and companion code must be submitted together.
6. Selecting the correct environment is important because the application depends on packages for numerical processing, images and checkpoint loading.
7. Running the inference demonstration should not start training.
8. The existing checkpoints are reused, and missing local assets are reported explicitly rather than replaced by fabricated examples.

## Chapter 10 of 30: Load sensor data

Save as: `10_Load_sensor_data.m4a` (or .wav / .mp3). About 1.0 min.

1. The sensor loading code examines known file structures rather than accepting every numerical table as interchangeable.
2. Supported CSV schemas identify expected channel names for Ottawa, UODS, MCC5, JUST and Aalto sources.
3. Original MAT recordings are also supported for relevant bearing sources.
4. CORA recordings have a source-specific naming and acquisition convention.
5. These checks matter because sampling rates, channel meanings and units affect the extracted features.
6. A pressure column cannot be substituted for an acceleration channel merely because both contain numbers.
7. The application rejects an unrecognized CSV schema instead of passing incompatible values into a model.
8. Source recognition is not a universal machine-identification algorithm; it identifies compatibility with the laboratory pipelines already implemented in this project.

## Chapter 11 of 30: Cleaning and feature preparation

Save as: `11_Cleaning_and_feature_preparation.m4a` (or .wav / .mp3). About 0.9 min.

1. Cleaning in this project means validating the measurements and applying the source-specific preparation expected by a model.
2. Readers check numeric values and whether enough samples are available for a complete feature window.
3. Later checks reject nonfinite features and feature sets that do not match the checkpoint.
4. These controls prevent a model from receiving the wrong columns in the wrong order.
5. Sensor features include quantities such as root mean square, standard deviation, crest factor and selected frequency-domain summaries.
6. Some sources use shaft-order information when rotational speed is available.
7. Preprocessing is not simply deleting missing rows and hoping for a good score.
8. Its purpose is to preserve a consistent relationship between the original measurement, the extracted features and the model that was trained on them.

## Chapter 12 of 30: Why the feature definitions matter

Save as: `12_Why_the_feature_definitions_matter.m4a` (or .wav / .mp3). About 0.9 min.

1. Root mean square is the square root of the average squared signal amplitude.
2. It summarizes signal energy but does not independently identify a particular fault.
3. Crest factor compares a peak amplitude with the root mean square value and can describe impulsive behaviour.
4. Spectral features summarize the distribution of energy across frequency.
5. In sources where speed is known, shaft-order features express frequency relative to rotation.
6. These measurements can change with operating conditions as well as damage, which explains why speed and load separation are important.
7. The model learns combinations of features from labelled examples.
8. A large feature value is not automatically a failure warning.
9. At inference, the selected checkpoint determines exactly which feature columns are required and which preprocessing pipeline must be applied.

## Chapter 13 of 30: Keep complete groups apart

Save as: `13_Keep_complete_groups_apart.m4a` (or .wav / .mp3). About 1.0 min.

1. The main evaluation safeguards are visible in the notebook split demonstration.
2. Complete recordings or operating groups are assigned to the test side.
3. Individual windows from a held-out recording cannot be placed in training.
4. Pump uses separate speed profiles or operating speeds for its two datasets.
5. CORA holds out whole speed settings.
6. Haul Truck sensor evaluation uses held-out bearing identities.
7. The aligned Motor Grader sensor experiment leaves gear pairs out, while its photo task uses source groups.
8. These protocols are not identical, and their differences remain part of the interpretation.
9. Inner validation selects models or training settings within the training data.
10. This reduces direct leakage from the final evaluation group, although dataset confounding and limited independent specimens can still restrict scientific conclusions.

## Chapter 14 of 30: Sensor model selection

Save as: `14_Sensor_model_selection.m4a` (or .wav / .mp3). About 1.0 min.

1. The sensor code defines several candidate classifiers rather than forcing a single algorithm onto every dataset.
2. The candidate family includes random forest, extra trees, histogram gradient boosting, a radial-basis support vector machine and logistic regression.
3. Scaling is contained inside the relevant estimator pipeline, so its parameters are fitted with training data.
4. Class weighting is used where configured.
5. The chosen candidate can differ across evaluation folds and across application mappings.
6. The notebook displays these settings by instantiating the candidate factories and reading their parameters.
7. Deployment uses the saved final estimator and its selected feature list.
8. It does not repeat candidate selection each time a user uploads a file.
9. This keeps inference reproducible and avoids confusing prediction latency with a new training experiment.

## Chapter 15 of 30: Image model inference

Save as: `15_Image_model_inference.m4a` (or .wav / .mp3). About 0.9 min.

1. The visual route reads a supported image, applies its configured transforms and uses a saved model to obtain probabilities.
2. Seven main image routes use ResNet18-based classifiers.
3. Motor Grader tooth photographs follow a separate path: frozen ResNet18 features feed a logistic-regression classifier.
4. This distinction is important when explaining which model produced an answer.
5. Inference runs without gradient calculation and with the model in evaluation mode.
6. The system is not trained to diagnose arbitrary whole-machine photographs.
7. Supported event frames, thermal frames, synthetic gear renders and tooth crops have different visual meanings and provenance.
8. A photo of an entire grader does not provide the same evidence as a labelled tooth inspection crop.
9. Unknown image sources must remain unsupported or ambiguous.

## Chapter 16 of 30: Data augmentation and selected models

Save as: `16_Data_augmentation_and_selected_models.m4a` (or .wav / .mp3). About 1.2 min.

1. Data augmentation is applied only to training images, never to validation or test images.
2. It imitates nuisance variation that should not change the fault label, and a seeded random generator keeps runs repeatable.
3. Event-camera and infrared frames receive random horizontal flips, shifts of up to eight pixels, brightness scaling between zero point eight and one point two, and small Gaussian noise.
4. Rendered gears are rotated by multiples of ninety degrees, mirrored and shifted by up to six pixels, because a gear looks the same when it is turned or reflected.
5. The gear-tooth photo route uses frozen features and is not augmented.
6. Sensor recordings are not augmented either; each recording already gives many windows, and class weighting handles imbalance.
7. Across the thirty-five outer evaluation rounds, inner validation selected logistic regression thirteen times and extra trees twelve times.
8. Gradient boosting was chosen five times, random forest three times and the RBF support vector machine twice.
9. No single classifier won every mapping, which is why the full candidate set is kept.

## Chapter 17 of 30: The central prediction function

Save as: `17_The_central_prediction_function.m4a` (or .wav / .mp3). About 0.9 min.

1. The central prediction function is the connection between the notebook, command line and local upload page.
2. It first handles the special tooth-image route when Motor Grader is explicitly selected.
3. Other inputs are passed through source identification.
4. If a user selects a mapping incompatible with that source, the function raises an error.
5. If the source matches several candidates and no reliable choice has been made, it returns an ambiguous equipment result.
6. Otherwise it scores the appropriate checkpoint, selects the most probable class and assembles the answer fields.
7. The common entry point reduces the risk that different interfaces accidentally apply different preprocessing.
8. It also keeps source identification separate from fault classification, which is essential for equipment mappings that share one laboratory dataset.

## Chapter 29 of 30: References and data sources

Save as: `29_References_and_data_sources.m4a` (or .wav / .mp3). About 1.0 min.

1. This chapter lists the main sources used in the project.
2. The sensor data come from the Ottawa and UODS bearing datasets, the CORA electromechanical dataset, the MCC5-THU gearbox data, the Aalto gear failure dataset and a slewing-bearing dataset.
3. The image data come from the XJTU dynamic-vision dataset, CORA infrared frames, MFGNet gear renders and the GearBox tooth photographs.
4. All nine public datasets are published under the Creative Commons Attribution 4.0 licence.
5. The models build on Breiman's random forests and on the residual networks of He and colleagues.
6. The mining context and the evaluation design draw on Dayo-Olupona and colleagues' review of predictive maintenance in mining, Zhao and colleagues' diagnosis benchmark, and Kaufman and colleagues' work on data leakage.
7. The final report lists all twenty-four references with full citations.
