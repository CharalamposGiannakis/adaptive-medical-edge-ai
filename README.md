# Adaptive Medical Edge AI

A course project exploring **adaptive inference at the edge**, with lung ultrasound classification as the current application scenario.

The current OpenPOCUS direction investigates **variable frame counts per lung-zone sequence prediction** as an adaptive input/workload mechanism. The earlier shared-transformer, intermediate-exit work is a feasibility pilot, not a final architecture choice. The project studies predictive performance, FLOPs, throughput, latency, adaptive policies and deployment constraints at the edge.

## Current status

The first feasibility experiment (**F001**) is complete.

We fine-tuned an ImageNet-pretrained **DeiT-Tiny** on **DermaMNIST (224×224)** and evaluated one shared backbone at transformer depths **4, 8, and 12**.

| Depth | Validation accuracy | Macro-F1 | fvcore FLOPs | Mean latency |
|---:|---:|---:|---:|---:|
| 4 | 74.48% | 0.3389 | 0.3792 G | 1.67 ms |
| 8 | 84.25% | 0.6655 | 0.7293 G | 3.25 ms |
| 12 | 84.45% | 0.6955 | 1.0794 G | 4.27 ms |

The depth-8 operating point is particularly promising: it nearly matches the full 12-block model in validation accuracy while using less reported compute and lower measured latency.

These are **validation-only feasibility results**. The official test split is intentionally untouched.

## Project direction

The current direction is to:

1. train a static OpenPOCUS proof of concept and characterize candidate frame-count operating points;
2. design an adaptive policy that selects measured input-workload operating points;
3. evaluate predictive performance, FLOPs and throughput;
4. study how the system behaves under edge constraints;
5. keep the design reproducible and deployable rather than treating adaptivity only as an offline model experiment.

The prehospital/ambulance lung POCUS use-case design and variable-frame direction were accepted in TA Sudaksh's feedback. OpenPOCUS remains a technical experimental proxy; T007 proposes bounded F002 frame counts, while the winning architecture, final adaptive operating points and controller mapping remain undecided (see D020–D024 in `docs/decision_log.md`).

The current conditional technical plan is in [`docs/design_pathway.md`](docs/design_pathway.md).
The proposed, preregistered F002 qualification and static-baseline contract is
in [`docs/f002_methodology.md`](docs/f002_methodology.md), with exact values in
[`configs/f002_protocol.json`](configs/f002_protocol.json). The copied
[`docs/adaptivity_project_notes.md`](docs/adaptivity_project_notes.md) contains
course-derived advisory notes; it does not override project decisions. No F002
model has trained or been selected yet.

## Repository structure

```text
adaptive-medical-edge-ai/
│
├── README.md
├── PROJECT_STATE.md
│
├── docs/
│   ├── decision_log.md
│   ├── open_tasks.md
│   ├── ideas.md
│   ├── experiment_log.md
│   ├── research_track.md
│   ├── presentation.md
│   └── ai_workflow.md
│
├── experiments/
│   └── feasibility_001/
│
├── src/
├── configs/
├── tests/
└── artifacts/
```

## Working style

The project is developed through GitHub with small, reproducible changes and explicit experiment tracking.

Basic Git/GitHub workflow is expected for collaboration:

```text
git pull
git status
git add .
git commit
git push
```

Branch-based work will be used when multiple people are developing in parallel.

## Datasets

Each dataset folder under `data/` holds its download script, a README and any small label files. The downloaded data goes into that folder's `raw/`, which is excluded from Git.

**OpenPOCUS** (lung-ultrasound video, general classification, no pneumothorax labels) is fetched with:

```bash
bash data/openpocus/download_openpocus.sh
```

This downloads `Lung Database.zip` (~5.3 GB) and unpacks the frames into `data/openpocus/raw/`. The labels are committed as `data/openpocus/labels.csv`; `data/openpocus/README.md` explains the label codes and dataset caveats. The dataset itself is CC BY-NC-SA 4.0 (non-commercial, share-alike); the accompanying paper is CC BY 4.0. Background on the dataset (device breakdown, labeling protocol, interrater reliability) is in `docs/OpenPOCUS_paper_Kumar2026.pdf` and `docs/OpenPOCUS_supplementary_S1.pdf`.

## Reproducibility

Experiment-specific setup and run instructions live inside each experiment directory. Generated datasets, checkpoints, artifacts, caches, and local environments are excluded from Git.

The first experiment can be found in:

```text
experiments/feasibility_001/
```

See its README for the exact methodology, commands, outputs, and measurement caveats.
