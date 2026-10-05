# Adaptive Medical Edge AI

A course project exploring **adaptive inference at the edge**, with lung ultrasound classification as the current application scenario.

The current OpenPOCUS direction develops **TinyViT-5M as an adaptive backbone**,
starting from variable frame counts per lung-zone sequence prediction and measuring
other bounded adaptive dimensions where useful. The project studies predictive
performance, FLOPs, throughput, latency, adaptive policies and deployment
constraints at the edge.

## Current status

The architecture and data foundation is complete, and T009 adaptive-dimension
characterization is next. The frozen static reference remains F002b MobileNetV2
(D025): K16 validation macro-F1 0.859, abnormal recall 0.750 and 5.11 GFLOPs.
Charalampos and Salo selected the F002c TinyViT-5M epoch-8 checkpoint as the
adaptive-system backbone (D026). That same checkpoint reached validation macro-F1
0.8602/0.8676/0.8737 at K4/K8/K16 without K-specific retraining. Its canonical
shared frozen copy is
[`checkpoints/f002c_tinyvit5m_epoch8.pt`](checkpoints/f002c_tinyvit5m_epoch8.pt).

These are validation development results, not clinical performance or a claim of
universal architecture superiority. The T006 held-out test partition remains
untouched.

## Project direction

The current direction is to:

1. characterize useful adaptive dimensions and costs around the selected TinyViT checkpoint;
2. construct at least three measured shared-model operating points;
3. integrate one compression technique and a resource-aware policy;
4. study how the system behaves under edge constraints;
5. compare the final adaptive system with the frozen MobileNetV2 static reference.

The prehospital/ambulance lung POCUS use-case design and variable-frame direction
were accepted in TA Sudaksh's feedback. OpenPOCUS remains a technical experimental
proxy. D025 continues to govern the MobileNetV2 static reference; D026 selects
TinyViT for adaptive development. Adaptive operating points, compression and
controller choices remain open. No held-out OpenPOCUS test results are claimed.

The current technical pathway is in [`docs/design_pathway.md`](docs/design_pathway.md).
The independently reviewed F002 qualification and static-baseline contract is
in [`docs/f002_methodology.md`](docs/f002_methodology.md), with exact values in
[`configs/f002_protocol.json`](configs/f002_protocol.json). The copied
[`docs/adaptivity_project_notes.md`](docs/adaptivity_project_notes.md) contains
course-derived advisory notes; it does not override project decisions. See the
historical records in [`experiments/F002/README.md`](experiments/F002/README.md)
and [`experiments/F002c/README.md`](experiments/F002c/README.md).

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
├── experiments/          # experiment-specific code and reproducibility records
│
├── checkpoints/          # explicitly tracked frozen handoffs
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

Experiment-specific setup and run instructions live inside each experiment
directory. Generated datasets, ordinary checkpoints, artifacts, caches, and local
environments are excluded from Git. The one tracked checkpoint exception is the
canonical frozen F002c adaptive backbone above; its provenance and archival
boundaries are recorded in
[`experiments/F002c/README.md`](experiments/F002c/README.md#canonical-shared-checkpoint).

The first experiment can be found in:

```text
experiments/feasibility_001/
```

See its README for the exact methodology, commands, outputs, and measurement caveats.
