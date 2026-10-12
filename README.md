# Adaptive Medical Edge AI

A course project exploring **adaptive inference at the edge**, with lung ultrasound classification as the current application scenario.

The current OpenPOCUS direction develops **TinyViT-5M as an adaptive backbone**,
starting from variable frame counts per lung-zone sequence prediction and measuring
other bounded adaptive dimensions where useful. The project studies predictive
performance, FLOPs, throughput, latency, adaptive policies and deployment
constraints at the edge.

## Current status

The architecture and data foundation is complete. T009 (F003) found frame count
to be the useful adaptive axis; T010 (F004) serves the TinyViT checkpoint at
K = 1–16 frames as one weight file; T011 (F005) adds nested structured pruning
with distillation, giving 4 widths × 6 K in one file. D027–D029 record these as
accepted decisions. T010 and T011 are complete; T012b's provisional
resource-aware controller was implemented and merged in PR #27. D030 proposes
10% simulated budget headroom for F007; Charalampos approved the direction,
and Salo's explicit policy approval remains pending. T013a now proposes the
[`F007 final-evaluation protocol`](experiments/F007/README.md). Team review
and acceptance of the numerical policy and that protocol must precede held-out
test evaluation (T013b). The frozen static reference remains F002b MobileNetV2
(D025): K16 validation macro-F1 0.859,
abnormal recall 0.750 and 5.11 GFLOPs.
Charalampos and Salo selected the F002c TinyViT-5M epoch-8 checkpoint as the
adaptive-system backbone (D026). That same checkpoint reached validation macro-F1
0.8602/0.8676/0.8737 at K4/K8/K16 without K-specific retraining. Its canonical
shared frozen copy is
[`checkpoints/f002c_tinyvit5m_epoch8.pt`](checkpoints/f002c_tinyvit5m_epoch8.pt).

These are validation development results, not clinical performance or a claim of
universal architecture superiority. The T006 held-out test partition remains
untouched.

## Project direction

The current pathway covers completed experiments and the remaining policy and
evaluation:

1. characterize useful adaptive dimensions and costs around the selected TinyViT checkpoint (T009, complete);
2. construct measured shared-model operating points (T010, complete);
3. integrate compute compression (T011, complete) and evaluate a provisional resource-aware policy (T012b implementation merged);
4. review how the system behaves under simulated edge constraints;
5. review the T013a protocol, then compare the final adaptive system with the frozen MobileNetV2 static reference (T013b, later).

The prehospital/ambulance lung POCUS use-case design and variable-frame direction
were accepted in TA Sudaksh's feedback. OpenPOCUS remains a technical experimental
proxy. D025 continues to govern the MobileNetV2 static reference; D026 selects
TinyViT for adaptive development. D027–D029 accept the frame-count family and
nested structured pruning as compute compression. Its shared checkpoint remains
about 20.4 MB; narrower widths reduce inference work. The T012b controller
policy remains provisional pending Salo's explicit approval of D030. F007 also
needs independent review and team approval. No held-out OpenPOCUS test results
are claimed.

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
│   ├── design_pathway.md
│   ├── experiment_log.md
│   ├── research_track.md
│   └── TEAM_WORKFLOW.md
│
├── experiments/          # experiment-specific code and reproducibility records
│
├── checkpoints/          # explicitly tracked frozen/deployable handoffs
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
environments are excluded from Git. Explicit canonical checkpoints are tracked
under `checkpoints/`. The F002c backbone's provenance and archival boundaries
are recorded in
[`experiments/F002c/README.md`](experiments/F002c/README.md#canonical-shared-checkpoint).
The frozen F002b static reference and the F005b deployable bundle are also
canonical tracked handoffs under [`checkpoints/`](checkpoints/); raw experiment
outputs remain ignored. The F005b bundle supports T012 without retraining.

The first experiment can be found in:

```text
experiments/feasibility_001/
```

See its README for the exact methodology, commands, outputs, and measurement caveats.
