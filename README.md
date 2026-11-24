# AI-Based Predictive Maintenance for Mining Equipment

**Group 9:** Ganesh Mishra, Lakshya Sharma and Sagar Patel
Saskatchewan Polytechnic

Upload a supported sensor file or component image to check its current fault condition. The application returns a fault label, model score and the model used.

We use public laboratory datasets as examples of components found in mining equipment. The project covers Pump, Haul Truck, Belt Conveyor, Dozer, Excavator, Wheel Loader, Drill Rig and Motor Grader. It has not been tested on operating mining machines and does not predict a failure date.

## Main files

| File | What it does |
|---|---|
| [Prediction notebook](Notebook/Integrated_Equipment_Fault_Prediction.ipynb) | Upload a file and get a prediction |
| [Experiment notebook](Notebook/Mining_Equipment_Condition_Monitoring.ipynb) | Review data preparation, models and results |
| [predictive_maintenance.py](Notebook/predictive_maintenance.py) | Data loading, training, prediction and local web page |
| [test_application.py](tests/test_application.py) | Application tests in one file |

## Folder layout

```text
Notebook/   two notebooks and the main Python file
tests/      one application test file
datasets/   source list, file manifests and a labelled synthetic example
docs/       project guide and original experiment records
Results/    saved scores
Figures/    result graphs
```

The example CSV is synthetic and is only for checking the file format. It is not research evidence.

## Run in VS Code

Install Python 3.11 and the VS Code Python and Jupyter extensions. Open this repository folder, then create an environment:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Mac/Linux, then install the packages:

```bash
python -m pip install -r requirements.txt
```

Open the prediction notebook, select the `.venv` kernel and run the cells. For the local upload page, run:

```bash
python Notebook/predictive_maintenance.py serve
```

Open `http://localhost:8613`. Stop the server with Ctrl+C.

## Data and model files

Real predictions need the source data, exported images and saved models on your computer. They are not downloaded automatically or included in Git. See the [dataset manifest](datasets/README.md) and the [project guide](docs/project_guide.md#running-the-application) for folders, trusted checkpoints and commands. The experiment notebook can display saved results without the raw datasets.

Some equipment uses the same dataset format. If the application cannot identify the mapping, it asks you to choose rather than guessing. The displayed score is a model output, not a guaranteed diagnosis.

## Results and documentation

Start with the [project guide](docs/project_guide.md). The [evidence matrix](docs/project_guide.md#section-equipment-evidence-matrix) lists the component, fault labels and scores for each mapping. [Results](Results/README.md) contains the saved metrics and [Figures](Figures/) contains the graphs.

Whole-group tests and time-split results are reported separately. Some models perform poorly at unseen speeds. CORA data are reused across mappings, Drill Rig images are synthetic, and sensor/image samples are not a fused dataset. The [methodology](docs/project_guide.md#section-methodology), [validation](docs/project_guide.md#section-scientific-validation) and [experiment archive](docs/experiment_archive.md) explain these limits. Use the [demo script](docs/project_guide.md#section-demo-script) to rehearse.

## Check the application

```bash
python -m pip install pytest pyflakes
python -m pyflakes Notebook tests
python -m pytest tests -q
python Notebook/predictive_maintenance.py route-check
```

The test suite skips two real-file tests when their local assets are missing. The route check needs the actual samples and checkpoints; successful execution is not a held-out accuracy score.

The code uses the [MIT License](LICENSE). Dataset licences and credits remain with their publishers. Reports, videos, raw datasets and model binaries stay local.
