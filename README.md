# Adaptive Medical Edge AI

A course project exploring **adaptive inference for Vision Transformers at the edge**, with medical image classification as the current application scenario.

The core idea is to use **one shared transformer backbone with intermediate exits**. Easy inputs may be classified after fewer transformer blocks, while harder inputs continue deeper through the network. The project studies the trade-off between predictive performance, computation, inference latency, adaptive decision policies, and deployment constraints at the edge.

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

1. establish useful operating points from a shared transformer;
2. design an adaptive policy that decides when an input should exit;
3. evaluate the accuracy–latency–compute trade-off;
4. study how the system behaves under edge constraints;
5. keep the design reproducible and deployable rather than treating adaptivity only as an offline model experiment.

The exact final scope is still open and will be refined after discussion with the course staff.

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

Each dataset folder under `data/` holds its download script. Only the script is committed; everything it creates is excluded from Git.

**OpenPOCUS** (lung-ultrasound video, general classification, no pneumothorax labels) is fetched with:

```bash
bash data/openpocus/download_openpocus.sh
```

This pulls `Lung Database.zip` (~5.3 GB) and the license file into `data/openpocus/`, unpacks the frames into `raw/`, and writes `labels.csv` plus a `README.md` explaining the label codes. The dataset itself is CC BY-NC-SA 4.0 (non-commercial, share-alike); the accompanying paper is CC BY 4.0. Background on the dataset (device breakdown, labeling protocol, interrater reliability) is in `docs/OpenPOCUS_paper_Kumar2026.pdf` and `docs/OpenPOCUS_supplementary_S1.pdf`.

## Reproducibility

Experiment-specific setup and run instructions live inside each experiment directory. Generated datasets, checkpoints, artifacts, caches, and local environments are excluded from Git.

The first experiment can be found in:

```text
experiments/feasibility_001/
```

See its README for the exact methodology, commands, outputs, and measurement caveats.
