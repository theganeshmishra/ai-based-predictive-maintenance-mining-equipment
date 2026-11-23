# Video recording kit: human narration

The current walkthrough video uses computer narration. This kit lets Ganesh, Lakshya and Sagar replace it with
their own voices. The slides, captions and chapter markers are then rebuilt to follow your recordings.

## What each person records

| Person | Script | Approx. reading time |
|---|---|---|
| Ganesh Mishra (team lead and developer) | [Script_Ganesh_Mishra.md](Script_Ganesh_Mishra.md) | 9 min |
| Lakshya Sharma (developer and tester) | [Script_Lakshya_Sharma.md](Script_Lakshya_Sharma.md) | 9 min |
| Sagar Patel (planner and developer) | [Script_Sagar_Patel.md](Script_Sagar_Patel.md) | 10 min |

Chapters are split by role: Ganesh presents the overview, results and conclusions; Sagar presents the data
pipeline and models; Lakshya presents the testing, executed outputs and evaluation checks.

| Chapter | Title | Presenter | Minutes |
|---|---|---|---|
| 1 | AI-Based Predictive Maintenance | Ganesh Mishra | 0.9 |
| 2 | Project team and roles | All three (one short file each) | 0.4 |
| 3 | The problem and objectives | Ganesh Mishra | 1.0 |
| 4 | Research background | Ganesh Mishra | 1.0 |
| 5 | Eight application mappings | Ganesh Mishra | 1.0 |
| 6 | Dataset provenance | Ganesh Mishra | 0.9 |
| 7 | System block diagram | Ganesh Mishra | 0.9 |
| 8 | Open the integrated notebook | Sagar Patel | 0.9 |
| 9 | Check the local checkpoints | Lakshya Sharma | 0.8 |
| 10 | Load sensor data | Sagar Patel | 1.0 |
| 11 | Cleaning and feature preparation | Sagar Patel | 0.9 |
| 12 | Why the feature definitions matter | Sagar Patel | 0.9 |
| 13 | Keep complete groups apart | Sagar Patel | 1.0 |
| 14 | Sensor model selection | Sagar Patel | 1.0 |
| 15 | Image model inference | Sagar Patel | 0.9 |
| 16 | Data augmentation and selected models | Sagar Patel | 1.2 |
| 17 | The central prediction function | Sagar Patel | 0.9 |
| 18 | Notebook upload interface | Lakshya Sharma | 0.9 |
| 19 | Executed Pump sensor example | Lakshya Sharma | 0.8 |
| 20 | Executed Pump image example | Lakshya Sharma | 0.9 |
| 21 | Executed Motor Grader gear sensor | Lakshya Sharma | 0.9 |
| 22 | Executed Motor Grader tooth image | Lakshya Sharma | 0.8 |
| 23 | Ambiguous equipment is a valid result | Lakshya Sharma | 0.9 |
| 24 | Invalid inputs and errors | Lakshya Sharma | 0.8 |
| 25 | Full route-check evidence | Lakshya Sharma | 0.9 |
| 26 | Independent experimental results | Ganesh Mishra | 0.9 |
| 27 | Time split versus held-out groups | Lakshya Sharma | 0.9 |
| 28 | Applications and remaining limits | Ganesh Mishra | 1.0 |
| 29 | References and data sources | Sagar Patel | 1.0 |
| 30 | Final demonstration summary | Ganesh Mishra | 1.0 |

## How to record

1. Use a quiet room. A phone voice-memo app, Windows Sound Recorder or a headset microphone is fine.
2. Hold the phone or microphone about 20 cm from your mouth and keep the distance the same for every chapter.
3. Record **one file per chapter**, named exactly as the script says (for example `03_The_problem_and_objectives.m4a`).
   `.m4a`, `.wav` and `.mp3` all work.
4. Read the numbered lines in order and leave about **one second of silence between lines**. Do not read the
   line numbers aloud.
5. Speak at a calm pace. Each chapter should take roughly the time shown in the table, but it does not need to match.
6. If you make a mistake, pause and re-read the whole line. Small stumbles can be left in; they sound natural.
7. Optional: record a short camera clip (landscape, 30 to 60 seconds) where all three of you say your name and role.
   Name it `00_camera_intro.mp4`. It will be placed before the slides.

## Send the files back

Put all files in one folder (for example `Downloads/Group9_Narration/`) and tell Claude where it is. The video will
then be rebuilt with:

- each slide timed to your voice, using the pauses between lines;
- captions and the transcript updated from what you actually say;
- the slide footer changed from "Computer narration" to "Team narration";
- any chapter without a recording kept in computer narration and listed, so nothing is silently mixed.

The numbers in the scripts match the final report. Please do not change figures while reading; if a line is hard
to say, tell Claude and it will be reworded.
