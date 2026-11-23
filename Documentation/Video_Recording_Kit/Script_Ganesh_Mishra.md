# Recording script: Ganesh Mishra

Role: Team lead and developer. About 9 minutes of reading in total.

Read [README.md](README.md) first. Save **one audio file per chapter**, named exactly as shown. Leave about one second of silence between numbered lines, so each slide can be matched to your voice. If you stumble, pause and re-read the whole line; do not stop the recording.

## Chapter 1 of 30: AI-Based Predictive Maintenance

Save as: `01_AI_Based_Predictive_Maintenance.m4a` (or .wav / .mp3). About 0.9 min.

1. Welcome to this technical walkthrough of AI-Based Predictive Maintenance for Mining Equipment.
2. We are Ganesh Mishra, Lakshya Sharma and Sagar Patel, Group 9 at Saskatchewan Polytechnic, and our supervisor is Doctor Himanshu Patel.
3. This video explains the objectives, data sources, software design, code and actual prediction outputs.
4. The execution examples were run earlier against the existing local checkpoints and are replayed here for readability.
5. The implemented task is current component fault classification.
6. Although predictive maintenance is the project title, the software does not estimate the date of a future failure.
7. That distinction will remain visible throughout the demonstration.

## Chapter 2 of 30: Project team and roles (your part)

Save as: `02_Project_team_and_roles_Ganesh.m4a` (or .wav / .mp3)

1. This chapter introduces our team.
2. I am Ganesh Mishra, the team lead and a developer.

## Chapter 3 of 30: The problem and objectives

Save as: `03_The_problem_and_objectives.m4a` (or .wav / .mp3). About 1.0 min.

1. Mining operations depend on components such as bearings, motors and geared drives.
2. A fault in one component may interrupt the larger process.
3. The project asks whether available condition-monitoring measurements can support a practical fault-classification prototype.
4. The software objective is straightforward: accept a supported file, apply the correct preprocessing, load an existing model, and display its predicted condition.
5. The research objective is harder: compare sensor and visual evidence using fault targets that are mechanically comparable, and test how results change under proper group separation.
6. A working upload button alone cannot establish that a classifier generalizes.
7. For that reason, the project keeps software execution checks separate from held-out experimental performance.
8. Both are necessary, but they answer different questions.

## Chapter 4 of 30: Research background

Save as: `04_Research_background.m4a` (or .wav / .mp3). About 1.0 min.

1. The research background combines conventional condition monitoring with visual transfer learning.
2. Sensor measurements can be summarized using statistical and spectral features, making tree ensembles, support vector machines and logistic regression useful candidates.
3. Random forests provide a practical baseline for mixed numerical features.
4. For visual inputs, the project uses ImageNet-pretrained ResNet18 representations.
5. Most visual routes fine tune a classifier, while the gear-tooth photograph route uses frozen features with logistic regression.
6. These algorithm choices do not remove weaknesses in the data.
7. A model may learn a recording, operating speed or camera setup instead of a transferable fault signature.
8. The central methodological choice is therefore the way training and testing are separated.
9. Dataset sources and model references are listed in the references chapter near the end of this video, and in the final report.

## Chapter 5 of 30: Eight application mappings

Save as: `05_Eight_application_mappings.m4a` (or .wav / .mp3). About 1.0 min.

1. The system contains eight mining-equipment application mappings: Pump, Haul Truck, Belt Conveyor, Dozer, Excavator, Wheel Loader, Drill Rig and Motor Grader.
2. These names describe intended applications of laboratory components.
3. They do not mean that every dataset was collected from the named mining machine.
4. Pump and Haul Truck use bearing-related evidence.
5. Conveyor and Dozer use geared-drive evidence, with Dozer reusing the conveyor benchmark.
6. Excavator and Wheel Loader use related drive and motor tasks.
7. Drill Rig uses gearbox data, while the aligned Motor Grader comparison concerns gear damage.
8. The original Motor Grader bearing route is also preserved as a separate task.
9. Reuse of the same rig or records is explicitly disclosed, so eight application rows are not presented as eight independent field trials.

## Chapter 6 of 30: Dataset provenance

Save as: `06_Dataset_provenance.m4a` (or .wav / .mp3). About 0.9 min.

1. The sensor sources include Ottawa and UODS bearing measurements, CORA electromechanical measurements, MCC5 gearbox vibration, Aalto gear measurements and the separate JUST bearing task.
2. Visual sources include XJTU dynamic-vision recordings, CORA infrared frames, synthetic MFGNet gear renders and real GearBox tooth photographs.
3. Event-camera frames are not ordinary colour photographs.
4. Synthetic renders are not camera measurements from damaged mining equipment.
5. Multiple frames, windows or tooth crops may come from one source group, so their count must not be confused with a count of independent machines.
6. The dataset manifest records the source mappings and available local files.
7. Raw data and large checkpoints stay outside the public code archive, and the local installation needs access to those prerequisites.

## Chapter 7 of 30: System block diagram

Save as: `07_System_block_diagram.m4a` (or .wav / .mp3). About 0.9 min.

1. The block diagram begins with a file supplied through the notebook, command line or local upload page.
2. The system validates the input and tries to identify a known source format.
3. If the source is compatible with more than one equipment mapping, it returns an ambiguity warning rather than silently guessing.
4. A sensor file is converted into the features expected by its checkpoint.
5. An image is transformed according to the visual route.
6. The saved model produces class probabilities, and the largest output determines the displayed condition.
7. Both modalities share the same prediction entry point, but they remain independent models.
8. There is no fusion of unrelated sensor and image samples.
9. The final output includes the application mapping, fault label, model probability and model used.

## Chapter 26 of 30: Independent experimental results

Save as: `26_Independent_experimental_results.m4a` (or .wav / .mp3). About 0.9 min.

1. The final report contains the independent experimental results.
2. Pump reached ninety-five point eight three percent sensor balanced accuracy and one hundred percent visual balanced accuracy under their respective recording protocols.
3. Those protocols and rigs differ, so the result does not establish that vision is universally better.
4. The CORA tasks showed weak transfer to unseen speeds.
5. Haul Truck had weaker visual recognition and a load-confounded ball-fault class.
6. Motor Grader's aligned gear sensor result was seventy-seven point seven eight percent balanced accuracy, while the tooth-photo result was eighty-two point five nine percent.
7. Its high raw photo accuracy is affected by class imbalance, and damaged-tooth recall is lower.
8. The poster shows these same held-out balanced accuracies, and the report gives the full numerical evidence.

## Chapter 28 of 30: Applications and remaining limits

Save as: `28_Applications_and_remaining_limits.m4a` (or .wav / .mp3). About 1.0 min.

1. The practical outcome is a working educational prototype for component inspection and reproducible model comparison.
2. It demonstrates data ingestion, feature preparation, trusted model reuse, ambiguity handling and prediction output through a common interface.
3. Its remaining limitations are scientific as well as operational.
4. Some datasets share records, image crops remain related, synthetic renders need validation against real photographs, and confidence values are not calibrated failure risks.
5. The code package also needs local data and checkpoints to run the full demonstration.
6. Before a real maintenance decision could rely on this system, additional work would need site-specific measurements, balanced operating conditions, independent machines and a clear assessment of missed-fault costs.
7. The present software should support investigation and explanation, not replace qualified engineering judgment.

## Chapter 30 of 30: Final demonstration summary

Save as: `30_Final_demonstration_summary.m4a` (or .wav / .mp3). About 1.0 min.

1. To conclude, the coding phase provides one integrated notebook backed by a reusable Python implementation.
2. It accepts supported sensor recordings or component images, checks source compatibility, loads an existing model and returns a fault label with model information and probability.
3. Where source evidence cannot distinguish equipment mappings, the system reports ambiguity rather than guessing.
4. Real local execution checks confirm that the supported routes run, including the aligned Aalto gear sensor and Motor Grader tooth-image paths.
5. The final report explains the actual held-out results, the supplementary time-split check and the unsuccessful cases.
6. The project is complete as a laboratory-based demonstration with scientific limitations.
7. Future work should improve independent data and validation before expanding claims about mining-field reliability or future failure prediction.
8. Thank you for watching.
