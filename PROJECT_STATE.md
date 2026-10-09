# Current phase

The architecture and data foundation is complete. T009/F003 established frame
count as the primary adaptive axis (D027). T010/F004 completed the shared
K1/K2/K4/K8/K12/K16 family on unchanged D026 weights (D028). T011/F005
completed nested structured pruning with distillation as the compute-compression
step (D029). D027–D029 are accepted. T012a is screening the deployable F005b
width × K configurations against provisional quality requirements before a
resource-aware controller is selected; T012b remains pending human approval.
F002b MobileNetV2 remains the frozen static comparison; D026 selects the F002c
TinyViT-5M checkpoint as the model to develop adaptively. No model has run on the
T006 held-out test partition.

## Frozen foundation

The scenario is resource-aware inference for prehospital/ambulance lung POCUS,
using OpenPOCUS as a technical experimental proxy rather than evidence of trauma,
ambulance-population or clinical-deployment effectiveness. The D019 task predicts
normal versus definite abnormal (B-lines and/or consolidation) for one exam × lung
zone sequence. Accuracy and macro-F1 are the primary classification metrics.

T006 fixes the cohort, labels, known-family/shared-recording-group-disjoint split
and evaluation units (D023). The retained snapshot has 1,762 sequences: 1,232
train, 265 validation and 265 test. The split protects against known dependencies,
but the released identifiers cannot prove biological-patient independence. T009
through T012 use train and validation only; the test partition remains reserved for
a later pre-specified final evaluation.

The frozen static reference is F002b MobileNetV2 (D025), seed 2026 at K16:

- duplicate-aware validation macro-F1 0.859 and accuracy 0.886;
- normal recall 0.944 and abnormal recall 0.750;
- 5.11 GFLOPs under the F002 convention;
- checkpoint SHA-256
  `5bad132633fd6188c0e449a75f73294612479bb1f0dc1820d8e6c459f7bdb823`.

Its checkpoint, protocol, results and historical decisions remain immutable. See
[`experiments/F002/README.md`](experiments/F002/README.md) for the full record.

## Adaptive backbone

Charalampos and Salo selected timm
`tiny_vit_5m_224.dist_in22k_ft_in1k` as the adaptive-system backbone on
2026-10-05 (D026). The canonical shared frozen copy is
[`checkpoints/f002c_tinyvit5m_epoch8.pt`](checkpoints/f002c_tinyvit5m_epoch8.pt)
and has SHA-256
`30b70734426e84f429b83011abca04511540149d1fc3e416274c6fbcb14853b2`.

The same checkpoint, without K-specific retraining, produced the following
duplicate-aware validation results at threshold 0.5:

| K | Macro-F1 | Accuracy | Normal recall | Abnormal recall |
|---:|---:|---:|---:|---:|
| 4 | 0.8602 | 0.8819 | 0.9101 | 0.8158 |
| 8 | 0.8676 | 0.8898 | 0.9270 | 0.8026 |
| 16 | 0.8737 | 0.8937 | 0.9213 | 0.8289 |

At K16, TinyViT exceeded the frozen MobileNetV2 reference on validation macro-F1,
accuracy and abnormal recall, while MobileNetV2 retained higher normal recall.
K4 and K8 also retained useful predictive quality with fewer input frames. This is
single-seed development evidence under different training recipes, not statistical
or universal superiority. F003 and F004 later measured TinyViT's cost, latency
and throughput under one convention. Full evidence and caveats are in
[`experiments/F002c/README.md`](experiments/F002c/README.md).

## Current objective — T012 resource-aware policy

The validation-only [`F006a`](experiments/F006a/README.md) analysis maps the
24 F005b configurations by macro-F1 and FLOPs. The follow-up
[`F006a_2`](experiments/F006a_2/README.md) screens all 24 against initial
accuracy, macro-F1 and abnormal-recall requirements anchored to frozen F002b
MobileNetV2, then compares compute among qualifying cells. These are empirical
research requirements, not clinically validated safety thresholds. The team
has not approved final thresholds, budget normalization, operating points or
controller rule; no T012b runtime policy has been implemented.

T009/F003 ([`experiments/F003/README.md`](experiments/F003/README.md)) found
frame count to be the useful axis of the D026 TinyViT (D027, accepted).
T010/F004 (2026-10-07, validation only;
[`experiments/F004/README.md`](experiments/F004/README.md)) turns it into one
deployable family. GPU: median over three GTX 1080 Ti hosts; CPU: one Xeon
E5-2630 v3 node, 4 threads; both model only (CPU times vary by up to ±25% between
jobs at K12/K16). The intended deployment target is
CPU-class edge hardware, where the FLOPs savings also save time; the desktop
GPU is latency-bound below K8.

| K | GFLOPs | Macro-F1 | Accuracy | AUROC | GPU ms | CPU ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.29 | 0.816 | 0.846 | 0.888 | 9.9 | 32 |
| 2 | 2.57 | 0.829 | 0.854 | 0.918 | 10.2 | 49 |
| 4 | 5.15 | 0.860 | 0.882 | 0.937 | 10.2 | 81 |
| 8 | 10.30 | 0.868 | 0.890 | 0.950 | 12.1 | 201 |
| 12 | 15.45 | 0.870 | 0.890 | 0.953 | 17.4 | 381 |
| 16 | 20.60 | 0.874 | 0.894 | 0.952 | 22.7 | 538 |

All points run the same weights of the D026 checkpoint through
`experiments/F004/family.py`. Nested frames let a clip move up in K by encoding
only the added frames; warm switching between K costs at most 0.13 ms, while the
first request at a new K after start-up costs up to about 150 ms (cuDNN tuning).
Only K1 and K2 are resolved below K16 on validation. Running every frame of each
clip scores 0.870 at 240 GFLOPs on average, no better than K16. Fine-tuning across
K (F004b) failed its pre-specified rule (three seeds also stay below it), so
the T010 family keeps D026's weights (D028). D029 makes the pruned seed-2026
descendant the deployable family for T012; D028 remains the uncompressed reference.

T011/F005 ([`experiments/F005/README.md`](experiments/F005/README.md)): INT8
dynamic quantization of all Linear layers keeps quality at every K and shrinks
the file 3.2× (20.4 → 6.4 MB), but leaves FLOPs unchanged and is not faster on
the CPU. Static INT8 (F005c) costs ranking quality and is slower still. F005b
prunes hidden channels as nested widths (1.0/0.75/0.5/0.25 → 1.29/1.10/0.91/0.72
GFLOPs per frame) with Taylor importance and distillation from D026. Over three
seeds the widths average macro-F1 0.853/0.857/0.856/0.846 over the five trained
K (D026: 0.849): about the unpruned quality at 29–44% fewer FLOPs per frame, in
one shared 20.4 MB weight file (D029, accepted). This is compute compression:
the narrower widths reduce inference FLOPs, not the shared file size. The 25%
width is the noisiest; for the
deployable seed 2026 it averages 0.839 and is 0.04 below D026 at K12/K16.
Individual front points are local peaks. Epochs were selected on validation, so
these are development numbers until T013.

## Guardrails

- Keep the T006 cohort, labels, train/validation/test split and evaluation units
  frozen.
- Keep the F002b MobileNetV2 checkpoint as the immutable static reference.
- Use the F002c epoch-8 TinyViT checkpoint as the adaptive backbone unless a later
  documented team decision changes direction.
- Do not use the T006 test partition during T009–T012 development.
- Measure final candidate operating points with a consistent FLOPs, runtime and
  throughput convention.
- Give each experiment an ID, reproducible configuration/commands, raw evidence,
  and a clear separation between measurements and interpretation.

## Next steps

T009 characterized adaptive dimensions → T010 built six shared-weight
frame-count operating points (F004) → T011 added nested pruned widths (F005) →
T012a reviews quality and cost across width × K points → T012b implements an
approved resource-aware policy → T013 performs the final static-versus-adaptive
evaluation on the test partition.
